import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LowMemCNN(BaseFeaturesExtractor):
    """
    A compressed CNN architecture designed for 64x64 visual RL on consumer GPUs.
    Input: (3, 64, 64) -> Output: (256,) vector
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # We assume input is 3 channels (RGB)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Layer 1: Aggresive downsampling (Stride 2)
            # Input: 64x64 -> Output: 31 x 31
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 2: Continued compression
            # Input: 31x31 -> Output: 14x14
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 3: Feature extraction
            # Input: 14x14 -> Output: 12x12
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size automatically to avoid math errors
        with torch.no_grad():
            # Create a dummy input to pass through the CNN
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            # We permute dimensions if necessary (Gym is usually CxHxW, but verify)
            # Standard Gym "rgb_array" is HxWxC, but SB3 wraps it to CxHxW automatically.
            # We trust SB3's 'CnnPolicy' handles the transpose.
            n_flatten = self.cnn(dummy_input).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Pass the observations through the CNN
        x = self.cnn(observations)

        # Flatten the output and pass through the linear layer
        x = self.linear(x)

        # Apply ReLU activation
        x = self.relu(x)

        return x


print("LowMemCNN Architecture definition loaded.")
