import gymnasium as gym
import matplotlib.pyplot as plt
import shimmy

# 1. We fix the ID to lowercase 'dm_control'
# 2. We move height/width into the 'render_kwargs' dictionary
env = gym.make(
    "dm_control/cheetah-run-v0",
    render_mode="rgb_array",
    render_kwargs={
        "height": 64,
        "width": 64,
        "camera_id": 0,  # 0 is usually the main tracking camera
    },
)

# Reset and render
obs, info = env.reset()
frame = env.render()

print(f" Environment Loaded! Frame shape: {frame.shape}")

# Visualize
plt.figure(figsize=(4, 4))
plt.imshow(frame)
plt.title("What your Agent Sees (64x64)")
plt.axis("off")
plt.show()

env.close()
