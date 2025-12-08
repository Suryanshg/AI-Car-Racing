import torch
import torch.nn as nn
from typing import Tuple

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling to get spatial map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(spatial))
        return x * attention

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out



# TODO: Look into implementing returning Normal Distribution mean and log_std instead of Beta Dist
class PPO_Network(nn.Module):

    def __init__(self, state_dim = (4, 96, 96), action_dim = 3):
        """
        TODO:

        Args:
            state_dim (tuple, optional): _description_. Defaults to (4, 96, 96).
            action_dim (int, optional): _description_. Defaults to 3.
        """
        super(PPO_Network, self).__init__()

        # Shared Convolutional Feature Extractor
        self.feature_extractor = nn.Sequential(
            ConvBlock(state_dim[0], 32, kernel_size=8, stride=4),
            SpatialAttention(kernel_size=3),

            ConvBlock(32, 64, kernel_size=4, stride=2),
            SpatialAttention(kernel_size=5),

            ConvBlock(64, 64, kernel_size=3, stride=1),
            SpatialAttention(kernel_size=7),

            nn.Flatten()
        )

        conv_out_size = 64 * 8 * 8

        # FC Layer for the Actor Head
        self.actor_fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU()
        )

        # Alpha & Beta Heads for the Actor
        self.actor_alpha = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus()
        )

        self.actor_beta = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softplus()
        )

        # FC Layer for Critic Head
        self.critic_fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, state) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        TODO:

        Args:
            state (_type_): _description_
        """

        # Extract Features using Convolution Layer
        conv_out = self.feature_extractor(state)

        # Predict the value of the state
        value = self.critic_fc(conv_out)

        # Extract Actor Features
        actor_features = self.actor_fc(conv_out)

        # Predict the alpha & beta for the policy distribution
        # Adding 1 to alpha & beta to make distribution "Concave & Unimodal"
        alpha = self.actor_alpha(actor_features) + 1
        beta = self.actor_beta(actor_features) + 1

        # Return the value of the state and the alpha and beta that describes the policy distribution
        return (alpha, beta), value
        

# Driver code to only print the architectures
if __name__ == '__main__':
    network = PPO_Network()
    print(network)

