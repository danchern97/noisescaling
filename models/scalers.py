import torch
import torch.nn as nn
import torch.distributions as D
from abc import ABC, abstractmethod
from typing import Literal
import torch.nn.functional as F
import torch.distributions as D


__all__ = ["StaticScaler", "StochasticScaler", "NoiseInjector", "GaussianMixtureScaler", "NormalizingFlowScaler", "SudokuNormalizingFlowScaler"]


class StaticScaler(nn.Module):
    """
    Generates fixed number of transformed representations of an input tensor by applying a list of transformations.
    This is a static ensemble baseline, as in Parallel Scalling Law (https://www.arxiv.org/abs/2505.10475).
    """

    def __init__(self, transformations: list[nn.Module]):
        """
        Initializes the StaticScaler.

        Args:
            transformations (list[nn.Module]): The transformations to apply to the input tensor.
        """
        super().__init__()
        self.transformations = nn.ModuleList(transformations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of shape (B, *) and returns a tensor of shape (B, n_reps, *).
        """
        return torch.stack([tr(x) for tr in self.transformations], dim=1)

class StochasticScaler(nn.Module, ABC):
    """
    Abstract base class for modules that modify an input tensor to produce
    multiple, varied, and stochastic representations of it.
    """

    def __init__(self, **kwargs):
        """
        Initializes the StochasticScaler.

        Args:
            **kwargs: Arguments (e.g. models, noise levels, etc.)
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, n_reps: int, **kwargs) -> torch.Tensor:
        """
        Takes a tensor and returns multiple modified versions of it, stacked along
        a new leading dimension.

        Args:
            x (torch.Tensor): The input tensor, e.g., of shape (B, *).
            n_reps (int): The number of representations to generate.

        Returns:
            x_hat (torch.Tensor): A tensor containing multiple representations,
                          e.g., of shape (B, n_reps, *).
        """
        raise NotImplementedError("This is an abstract class")
    
    @abstractmethod
    def get_distribution(self, x: torch.Tensor, **kwargs) -> list[D.Distribution]:
        """
        Computes the distribution of the stochastic representations.

        Args:
            x (torch.Tensor): The input tensor of shape (B, *).

        Returns:
            list[D.Distribution]: The distributions of the stochastic representations, parameterized by the input tensor.
        """
        raise NotImplementedError("This is an abstract class")
    

    def get_entropy(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Estimates the entropy using Monte Carlo sampling.

        This is required because the entropy of a stochastic scaler does not have a
        closed-form solution. The estimate is differentiable.

        Args:
            x (torch.Tensor): The input tensor.
            n_samples (int): The number of samples to use for the Monte Carlo estimation.

        Returns:
            torch.Tensor: An estimate of the entropy for each sample in the batch.
        """
        dist = self.get_distribution(x)
        
        # To get a differentiable estimate of the entropy, we must sample using
        # the reparameterization trick.
        samples = self.forward(x, n_reps=n_samples)
        
        # The entropy is the negative expectation of the log probability of samples
        # from the distribution. We estimate this with a Monte Carlo average.
        log_probs = dist.log_prob(samples) # (B, n_samples)

        entropy_estimate = -torch.mean(log_probs, dim=1) # (B,)
        
        return entropy_estimate


class NoiseInjector(StochasticScaler):
    """
    A baseline example of noise injection with only parameter of noise level (i.e. standard deviation) of the distribution.
    """

    def __init__(self, noise_level: torch.Tensor = torch.tensor(0.0, dtype=torch.float32, requires_grad=True), noise_type: Literal["uniform", "normal"] = "uniform"):
        """
        Initializes the NoiseInjector.

        Args:
            noise_level (float): The standard deviation of the Gaussian noise to be added.
            noise_type (Literal["uniform", "normal"]): The type of noise to be added.
        """
        super().__init__()
        self.noise_level = nn.Parameter(noise_level)
        self.noise_type = noise_type

    def forward(self, x: torch.Tensor, n_reps: int, **kwargs) -> torch.Tensor:
        """
        Adds uniform noise to the input tensor `n_reps` times during training. During
        evaluation, it returns identical copies.

        Args:
            x (torch.Tensor): The input tensor of shape (B, *).
            n_reps (int): The number of representations to generate.

        Returns:
            torch.Tensor: A tensor with shape (B, n_reps, *).
        """
        expanded_x = x.unsqueeze(1).expand(x.shape[0], n_reps, *x.shape[1:])  # (B, n_reps, *)

        if self.training and self.noise_level > 0:
            if self.noise_type == "uniform":
                noise = torch.rand_like(expanded_x) * self.noise_level
            elif self.noise_type == "normal":
                noise = torch.randn_like(expanded_x) * self.noise_level
            else:
                raise ValueError(f"Invalid noise type: {self.noise_type}")
            return expanded_x + noise  # (B, n_reps, *)
        
        return expanded_x  # (B, n_reps, *)
    
    def get_distribution(self, x: torch.Tensor) -> list[D.Distribution]:
        """
        Returns a list of distributions, one for each representation.
        """
        return [D.Uniform(0, self.noise_level.detach()) if self.noise_type == "uniform" else D.Normal(0, self.noise_level.detach()) for _ in range(x.shape[1])]
    
    def get_entropy(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the entropy of the stochastic representations.
        """
        return torch.tensor(0.0) if self.noise_type == "uniform" else (0.5 * torch.log(torch.tensor(2.0) * torch.pi * torch.e * self.noise_level**2)).repeat(x.shape[0])


class GaussianMixtureScaler(StochasticScaler):
    """
    Represents a Gaussian Mixture Model where the parameters are predicted by neural networks.
    The mixture is the mixture of Gaussians, where each Gaussian is multivariate normal with diagonal covariance matrix.

    This class takes three separate models to predict the mixture weights (logits),
    means, and log variances of the Gaussian components based on an input tensor `x`.

    It provides two main methods for interacting with the distribution:
    - `get_distribution()`: Returns a standard PyTorch `MixtureSameFamily` distribution object,
      suitable for calculating log-probabilities (`log_prob`).
    - `rsample()`: Provides a differentiable sample from the mixture using the reparameterization
      trick with Gumbel-Softmax, allowing gradients to be backpropagated through the sampling process.
    - `entropy()`: Provides a differentiable Monte Carlo estimate of the distribution's entropy.
    """

    def __init__(
        self,
        n_gaussians: int,
        mean_model: nn.Module,
        variance_model: nn.Module,
        weights_model: nn.Module,
    ):
        """
        Initializes the GaussianMixture model.

        Args:
            n_gaussians (int): The number of Gaussian components in the mixture.
            mean_model (nn.Module): A model that takes an input `x` and outputs the means for each Gaussian.
                                    The output tensor shape is expected to be (batch_size, n_gaussians, [data_dim]).
            variance_model (nn.Module): A model that takes an input `x` and outputs the log variances for each Gaussian.
                                        This is for numerical stability, as variance must be positive.
                                        The output tensor shape is expected to be (batch_size, n_gaussians, [data_dim]).
            weights_model (nn.Module): A model that takes an input `x` and outputs the logits for the mixture weights.
                                       The output tensor shape is expected to be (batch_size, n_gaussians).
        """
        super().__init__()
        self.n_gaussians = n_gaussians
        self.mean_model = mean_model
        self.variance_model = variance_model
        self.weights_model = weights_model

    def get_distribution_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the parameters of the GMM distribution for the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor for the component models of shape (B, *).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The parameters of the GMM distribution.
        """
        mixture_logits = self.weights_model(x) # (B, n_gaussians)
        means = self.mean_model(x) # (B, n_gaussians, *)
        log_variances = self.variance_model(x) # (B, n_gaussians, *)
        # We use scale which is \sigma = sqrt(variance) = exp(0.5 * log_variance) for stability
        scale = torch.exp(0.5 * log_variances)
        return mixture_logits, means, scale

    def get_distribution(self, x: torch.Tensor) -> D.MixtureSameFamily:
        """
        Creates a PyTorch distribution object representing the Gaussian Mixture.

        This is useful for sampling from the mixture or calculating log probabilities.

        Args:
            x (torch.Tensor): The input tensor for the component models of shape (B, *).

        Returns:
            torch.distributions.MixtureSameFamily: The GMM distribution object.
        """
        mixture_logits, means, scale = self.get_distribution_parameters(x)

        # Categorical distribution for the mixture weights
        mix = D.Categorical(logits=mixture_logits)

        # Component distributions (Normal)
        comp = D.Independent(D.Normal(loc=means, scale=scale), len(x.shape) - 1)

        # The complete mixture distribution
        gmm_distribution = D.MixtureSameFamily(mix, comp)

        return gmm_distribution

    def forward(self, x: torch.Tensor, n_reps: int, temperature: float = 1.0) -> torch.Tensor:
        """
        Samples from the GMM using the reparameterization trick with Gumbel-Softmax.

        This allows gradients to be backpropagated through the sampling process.

        Args:
            x (torch.Tensor): The input tensor for the component models of shape (B, *).
            n_reps (int): The number of representations to generate.
            temperature (float, optional): The temperature for the Gumbel-Softmax distribution.
                                     A lower temperature makes samples closer to one-hot. Defaults to 1.0.

        Returns:
            torch.Tensor: A differentiable sample from the GMM of shape (B, n_reps, *)
        """
        # Get parameters of the GMM
        mixture_logits, means, scale = self.get_distribution_parameters(x)

        # Get Gumbel-Softmax samples for component selection
        component_selection_logits = mixture_logits.unsqueeze(1).expand(-1, n_reps, -1)  # (B, n_reps, n_gaussians)
        component_selection = F.gumbel_softmax(component_selection_logits, tau=temperature, hard=True, dim=-1)
        expanded_shape = (*component_selection.shape, *((1,)*(len(x.shape) - 1)))
        component_selection = component_selection.view(expanded_shape) # (B, n_reps, n_gaussians, *)

        # Get samples from component distributions
        comp = D.Independent(D.Normal(loc=means, scale=scale), len(x.shape) - 1)
        component_samples = comp.rsample(sample_shape=torch.Size([n_reps])).swapaxes(0, 1)  # (B, n_reps, n_gaussians, *)

        # Combine samples based on Gumbel-Softmax selection
        gmm_samples = torch.sum(component_selection * component_samples, dim=2)  # (B, n_reps, *)

        return gmm_samples
class SimpleCouplingLayer(nn.Module):
    """
    A single affine coupling layer for a normalizing flow.

    This layer splits the input channels into two groups using a binary mask.
    One group is left unchanged, while the other is transformed using an affine transformation
    whose parameters are predicted by a neural network conditioned on the unchanged group.

    The mask alternates between layers to ensure all channels are transformed across the stack.

    Args:
        c_in (int): Number of input channels.
        hidden_dim (int): Number of hidden units in the coupling network.
        mask_even (bool): If True, mask even channels; if False, mask odd channels.
        n_layers (int): Number of layers in the coupling network.
    """
    def __init__(self, c_in, hidden_dim=64, mask_even=True, n_layers=3):
        super().__init__()

        layers = []
        for i in range(n_layers):
            in_ch = c_in if i == 0 else hidden_dim
            layers.append(nn.Conv2d(in_ch, hidden_dim, 3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_dim, 2 * c_in, 3, padding=1))
        self.net = nn.Sequential(*layers)

        mask = torch.zeros(1, c_in, 1, 1)
        if mask_even:
            mask[:, ::2] = 1
        else:
            mask[:, 1::2] = 1
        self.register_buffer('mask', mask)

    def forward(self, z, reverse=False):
        mask = self.mask
        z1 = z * mask
        h = self.net(z1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            z2 = (z * (1 - mask) + t) * torch.exp(s * (1 - mask))
        else:
            z2 = (z * (1 - mask)) * torch.exp(-s * (1 - mask)) - t
        return z1 + z2

class NormalizingFlowScaler(StochasticScaler):
    """
    Stochastic scaler using a stack of affine coupling layers as a normalizing flow.

    This module generates multiple stochastic, invertible transformations of an input tensor
    by sampling from a base distribution and passing the result through a sequence of
    SimpleCouplingLayers. The flow is designed to increase the diversity and expressivity
    of latent representations for downstream tasks.

    Args:
        n_coupling_layers (int): Number of coupling layers in the flow.
        hidden_dim (int): Number of hidden units in each coupling layer's network.
        n_reps (int): Number of stochastic representations to generate per input.
        n_coupling_net_layers (int): Number of layers in each coupling network.
        **kwargs: Additional arguments for the base class.
    """

    def __init__(self, n_coupling_layers=4, hidden_dim=64, n_reps=2, n_coupling_net_layers=3, **kwargs):
        super().__init__()
        self.n_reps = n_reps
        self.c_in = 256  

        self.flow_layers = nn.ModuleList([
            SimpleCouplingLayer(self.c_in, hidden_dim, mask_even=(i % 2 == 0), n_layers=n_coupling_net_layers)
            for i in range(n_coupling_layers)
        ])
        self.base_dist = D.Normal(0, 1)

    def forward(self, x: torch.Tensor, n_reps: int = None, **kwargs) -> torch.Tensor:
        n_reps = n_reps or self.n_reps
        B, C, H, W = x.shape
        z = self.base_dist.sample((B, n_reps, C, H, W)).to(x.device)
        z = z + x.unsqueeze(1)
        z = z.view(B * n_reps, C, H, W)
        for flow in self.flow_layers:
            z = flow(z)
        z = z.view(B, n_reps, C, H, W)
        return z

    def get_distribution(self, x: torch.Tensor, **kwargs):
        """
        Not implemented for NormalizingFlowScaler.
        """
        raise NotImplementedError("get_distribution is not implemented for NormalizingFlowScaler.")