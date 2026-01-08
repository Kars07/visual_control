DrQ-Nano: Visual Control on Consumer Hardware

Deep Reinforcement Learning from raw pixels on a 6GB RTX 4050.

Standard visual reinforcement learning (RL) systems typically require large-scale compute resources to learn policies from high-dimensional pixel observations. This project demonstrates that a Data-Regularized Q-Learning (DrQ) inspired pipeline, combined with Soft Actor-Critic (SAC), can solve pixel-based continuous control tasks entirely on consumer-grade hardware.

The agent is trained on the DeepMind Control Suite (Cheetah Run) using a constrained GPU memory budget.

Research Goals

Sample Efficiency
Can continuous control tasks from raw pixels be solved with fewer than 500k environment interactions?

Hardware Constraints
Can a visual CNN and off-policy replay buffer fit within 6GB of GPU VRAM?

Representation Learning
Can simple image-based data augmentation (random shifts) replace large-scale visual pre-training?

Architecture Overview
Low-Memory CNN (LowMemCNN)

Standard visual encoders (e.g., NatureCNN) are too memory-intensive when combined with large replay buffers. To address this, a compressed feature extractor was designed.

Input:

64 x 64 x 3 RGB images

Structure:

Three strided convolutional layers

ReLU activations

Output:

256-dimensional latent feature vector

Optimization:

Automatic convolution output size inference to avoid tensor shape mismatch errors

DrQ Augmentation Wrapper

To reduce overfitting to static backgrounds, the agent never observes the raw frame directly. Instead, each observation is randomly shifted by ±4 pixels along the horizontal and vertical axes.

This forces the policy to focus on task-relevant visual features rather than background artifacts.

Stack

Environment:

DeepMind Control Suite (via Shimmy and MuJoCo)

Algorithm:

Soft Actor-Critic (SAC) using Stable Baselines3

Optimizations:

Mixed precision training (FP16)

Gradient checkpointing

Installation

Requires MuJoCo support.

pip install gymnasium[mujoco] shimmy[dm-control] stable-baselines3 opencv-python matplotlib


Train the agent:

python train_agent.py

Results

Hardware:

NVIDIA RTX 4050 (Laptop), 6GB VRAM

Training Time:

Approximately 2 hours

Performance:

Solved DeepMind Control Suite "Cheetah Run"

Reward greater than 300

Converged in approximately 150k environment steps

Lessons Learned

Gymnasium vs. Gym
API-breaking changes required careful handling of the PixelObservationWrapper.

VRAM Management
Reducing the batch size to 64 and replay buffer size to 50k was critical to avoid out-of-memory errors on a 6GB GPU.

Latent Visual Dynamics
The agent learned to ignore static background textures and focus exclusively on limb motion and dynamics inferred from pixel gradients.

Credits

Algorithm:

Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels
Kostrikov et al., 2020

Libraries:

Stable Baselines3

Gymnasium

Shimmy

MuJoCo
