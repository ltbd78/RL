import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit
import argparse
import traceback
import pickle
import os

from RL.utils.render import *
from RL.agents import *


parser = argparse.ArgumentParser() # TODO


def test(pkl_path, pth_path, env, attempts, display=False, video_dir=None):
    with open(pkl_path, 'rb') as f:
        logs = pickle.load(f)

    if video_dir:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = RecordVideo(env, video_dir)

    if logs['agent'] == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space, **logs['params'])
        agent.epsilon = 0
    elif logs['agent'] == 'a2c':
        agent = A2CAgent(env.observation_space, env.action_space, **logs['params'])
    elif logs['agent'] == 'td3':
        agent = TD3Agent(env.observation_space, env.action_space, **logs['params'])
    elif logs['agent'] == 'random':
        agent = RandomAgent(env.observation_space, env.action_space, **logs['params'])

    agent.load(pth_path)

    try:
        rewards = []
        for attempt in range(attempts):
            state, info = env.reset()
            sum_reward = 0
            t = 0
            done = False
            while not done:
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                sum_reward += reward
                t += 1
                if display:
                    title = f'Attempt: {attempt+1} | Timestep: {t} | Reward: {reward} | Sum Reward: {sum_reward}'
                    render(env, title)
            rewards.append(sum_reward)
        env.close()
        return rewards
    except Exception:
        traceback.print_exc()
        breakpoint()
        env.close()
