import torch
import torch.nn as nn
from typing import Tuple

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
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size = 8, stride = 4),       # (N, 32, 23, 23)
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2),                 # (N, 64, 10, 10)
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size = 3, stride = 1),                 # (N, 64, 8, 8)
            nn.ReLU(),

            nn.Flatten()                                                    # (N, 64 * 8 * 8)
        )

        # TODO: Instead of hard coding, use a method to dynamically compute this
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
        conv_out = self.conv(state)

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
        

