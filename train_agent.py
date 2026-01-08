import gymnasium as gym
import shimmy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from custom_architecture import LowMemCNN, RandomShiftWrapper


# Setup Environment
def make_env():
    env = gym.make(
        "dm_control/cheetah-run-v0",
        render_mode="rgb_array",
        render_kwargs={"height": 64, "width": 64, "camera_id": 0},
    )
    env = RandomShiftWrapper(env, pad=4)
    return env


# Vectorize environment (SB3 expects a vectorized env for efficiency)
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)  # Ensures shape is (Channels, Height, Width) for PyTorch

# Configure the SAC Agent for 6gb VRAM
policy_kwargs = dict(
    features_extractor_class=LowMemCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=True,  # Divides pixel values by 255.0
    net_arch=[256, 256],  # Policy/Value network size
)

print("Initializing Agent...")
model = SAC(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    buffer_size=50000,  # REDUCED from 1M to 50k to save RAM
    batch_size=64,  # REDUCED from 256 to 64 for VRAM safety
    learning_rate=3e-4,
    learning_starts=1000,  # Start learning after 1000 frames of exploration
    train_freq=4,  # Update every 4 steps (efficiency)
    gradient_steps=1,
    verbose=1,
    device="cuda",
)

# 3. Train!
print("Starting Training....")
# We aim for 200,000 steps to see results (takes ~1-2 hours on 4050)
model.learn(total_timesteps=200_000, log_interval=100)

# 4. Save
model.save("drq_cheetah_agent")
print("Model Saved!")
