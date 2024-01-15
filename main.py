import gymnasium as gym

env = gym.make("LunarLander-v2", render_mode="human")
env.reset()

print('sample_action',env.action_space)
# env.action_space.seed(42)
# observation, info = env.reset(seed=42)
