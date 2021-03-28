import gym
from gym.wrappers import Monitor, TimeLimit
import argparse
import traceback
import numpy as np
import pickle
import os

from RL.utils.render import *
from RL.agents import *


parser = argparse.ArgumentParser() # TODO


def test(pkl_path, pth_path, env, attempts, display=False, video_dir=None): 
    with open(pkl_path, 'rb') as f:
        logs = pickle.load(f)
        
    env = TimeLimit(env, max_episode_steps=logs['params']['max_steps'])
    if video_dir:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        env = Monitor(env, video_dir, force=True)
        
    if logs['agent'] == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space, **logs['params'])
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
            state = env.reset()
            sum_reward = 0
            t = 0 
            done = False
            
            while not done:
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                sum_reward += reward
                t += 1
                
                if display:
                    title = f'Attempt: {attempt+1} | Timestep: {t} | Reward: {reward} | Sum Reward: {sum_reward}'
                    render(env, title)

            rewards.append(sum_reward)
                
        print('rewards:', rewards)
        print('mean(rewards) =', np.mean(rewards))
        print('-'*13)
        env.close()
    except Exception:
        traceback.print_exc()
        env.close()