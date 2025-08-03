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


class ConcatELU(nn.Module):
    def forward(self, x):
        return torch.cat([F.elu(x), F.elu(-x)], dim=1)

class LayerNormChannels(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, c_in, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, c_in, 1, 1))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        y = (x - mean) / torch.sqrt(var + self.eps)
        y = y * self.gamma + self.beta
        return y

class GatedConv(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.net = nn.Sequential(
            ConcatELU(),
            nn.Conv2d(2*c_in, c_hidden, 3, padding=1),
            ConcatELU(),
            nn.Conv2d(2*c_hidden, 2*c_in, 1)
        )
    def forward(self, x):
        out = self.net(x)
        val, gate = out.chunk(2, dim=1)
        return x + val * torch.sigmoid(gate)

class GatedConvNet(nn.Module):
    def __init__(self, c_in, c_hidden=64, c_out=-1, num_layers=2):
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = [nn.Conv2d(c_in, c_hidden, 3, padding=1)]
        for _ in range(num_layers):
            layers += [GatedConv(c_hidden, c_hidden), LayerNormChannels(c_hidden)]
        layers += [ConcatELU(), nn.Conv2d(2*c_hidden, c_out, 3, padding=1)]
        self.nn = nn.Sequential(*layers)
        nn.init.zeros_(self.nn[-1].weight)
        nn.init.zeros_(self.nn[-1].bias)
    def forward(self, x):
        return self.nn(x)

class SimpleCouplingLayer(nn.Module):

    def create_checkerboard_mask(h, w, invert=False):
        x, y = torch.arange(h), torch.arange(w)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        mask = ((xx + yy) % 2).float().view(1, 1, h, w)
        if invert:
            mask = 1 - mask
        return mask

    def __init__(self, c_in, hidden_dim=64, mask=None, mask_even=True, n_layers=2):
        super().__init__()
        self.net = GatedConvNet(c_in, hidden_dim, 2 * c_in, n_layers)
        if mask is not None:
            self.register_buffer('mask', mask.clone()) 
        else:
            mask = torch.zeros(1, c_in, 1, 1)
            if mask_even:
                mask[:, ::2] = 1
            else:
                mask[:, 1::2] = 1
            self.register_buffer('mask', mask)

    def forward(self, z, reverse=False, return_log_det=False):
        mask = self.mask
        z1 = z * mask
        h = self.net(z1)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s)
        if not reverse:
            z2 = (z * (1 - mask) + t) * torch.exp(s * (1 - mask))
            log_det = (s * (1 - mask)).view(z.shape[0], -1).sum(-1)
        else:
            z2 = (z * (1 - mask)) * torch.exp(-s * (1 - mask)) - t
            log_det = -(s * (1 - mask)).view(z.shape[0], -1).sum(-1)
        out = z1 + z2
        if return_log_det:
            return out, log_det
        else:
            return out

class NormalizingFlowScaler(StochasticScaler):

    def __init__(self, n_coupling_layers=4, hidden_dim=64, n_reps=2, n_coupling_net_layers=3, H=9, W=9, **kwargs):
        super().__init__()
        self.n_reps = n_reps
        self.c_in = 256  # adjust as needed

        self.flow_layers = nn.ModuleList()
        for i in range(n_coupling_layers):
            # Alternate checkerboard mask (invert every other layer)
            mask = SimpleCouplingLayer.create_checkerboard_mask(H, W, invert=(i % 2 == 1))
            mask = mask.to(torch.float32).to('cpu')  # or .to(device) if needed
            # Expand mask to all channels
            mask = mask.expand(1, self.c_in, H, W).clone()
            self.flow_layers.append(
                SimpleCouplingLayer(self.c_in, hidden_dim, mask=mask, n_layers=n_coupling_net_layers)
            )
        self.base_dist = D.Normal(0, 1)

    def forward(self, x: torch.Tensor, n_reps: int = None, **kwargs):
        n_reps = n_reps or self.n_reps
        B, C, H, W = x.shape
        z = self.base_dist.sample((B, n_reps, C, H, W)).to(x.device)
        z = z + x.unsqueeze(1)
        z = z.view(B * n_reps, C, H, W)
        sum_log_det = torch.zeros(z.shape[0], device=z.device)
        for flow in self.flow_layers:
            z, log_det = flow(z, return_log_det=True)
            sum_log_det += log_det
        z = z.view(B, n_reps, C, H, W)
        sum_log_det = sum_log_det.view(B, n_reps)
        return z, sum_log_det

    def get_distribution(self, x: torch.Tensor, **kwargs):
        """
        Not implemented for NormalizingFlowScaler.
        """
        raise NotImplementedError("get_distribution is not implemented for NormalizingFlowScaler.")