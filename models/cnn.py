
import torch.nn as nn
import torch.nn.functional as F

########################################################
# SudokuCNN
########################################################

# This Sudoku CNN is based on the following post: https://www.kaggle.com/code/lyly123/solve-sudoku-with-cnn-acc-97

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 9, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, 81 * 9),
            nn.ReLU(),
            nn.LayerNorm(81 * 9),
            Reshape((-1, 9, 9, 9)),
        )

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)