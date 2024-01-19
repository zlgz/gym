import gymnasium as gym
# 其中A2C是value-based和policy-based的混合体
from stable_baselines3 import A2C,PPO


env = gym.make('LunarLander-v2', render_mode='human')

# 首先实例化模型，
# model = A2C('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', env, verbose=1)
# 学习100个时间步
model.learn(total_timesteps=100)

episodes = 10
# 可视化环境官网命名为vec_env
vec_env = model.get_env()
obs = vec_env.reset()

for ep in  range(episodes):
    # 用来定义游戏是否结束
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = vec_env.step(action)
        env.render()
        print(rewards)
env.close()



# env.reset()
#
# # 落地之后没有结束
# for step in range(200):
#     env.render()
#
#     # env.action_space.sample()指的是随机动作
#     observation, reward, terminated, truncated, info=env.step(env.action_space.sample())
#     print(reward,terminated)
#     # 根据terminated的值来确定是否游戏重新开始
