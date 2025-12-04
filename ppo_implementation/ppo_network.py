import torch
import torch.nn as nn
from typing import Tuple

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
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size = 8, stride = 4),       # (N, 32, H//4, W//2)
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
            # nn.LazyLinear(256),
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
            # nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


        # Initialize weights
        self._init_weights()


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
    

    def _init_weights(self):
        # Orthogonal init for conv layers with ReLU gain
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        # Orthogonal init for actor/critic hidden layers
        for m in list(self.actor_fc) + list(self.critic_fc):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

        # Actor output layers: smaller std helps stabilize policy updates
        for head in [self.actor_alpha, self.actor_beta]:
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.constant_(m.bias, 0.0)

        # Critic output: gain=1 for value regression
        last_critic = self.critic_fc[-1]
        if isinstance(last_critic, nn.Linear):
            nn.init.orthogonal_(last_critic.weight, gain=1.0)
            nn.init.constant_(last_critic.bias, 0.0)
        

# Driver code to only print the architectures
if __name__ == '__main__':
    network = PPO_Network()
    print(network)

