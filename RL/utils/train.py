import gym
from gym.wrappers import TimeLimit
import pickle
import argparse
import traceback
import os

from RL.agents import *


parser = argparse.ArgumentParser() # TODO


def train(agent_type, env, verbose=True, save_freq=50, save_dir='.', **params):
    if verbose:
        print(params)
    
    if agent_type == 'dqn':
        agent = DQNAgent(env.observation_space, env.action_space, **params)
    elif agent_type == 'a2c':
        agent = A2CAgent(env.observation_space, env.action_space, **params)
    elif agent_type == 'td3':
        agent = TD3Agent(env.observation_space, env.action_space, **params)
    elif agent_type == 'random':
        agent = RandomAgent(env.observation_space, env.action_space, **params)
    
    env = TimeLimit(env, max_episode_steps=params['max_steps'])
    log = {'agent':agent_type, 'params':params, 'episodes':[]}
 
    model_dir = save_dir + 'models/' + agent_type + '/'
    log_dir = save_dir + 'logs/' + agent_type + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    try:
        for e in range(params['max_episodes']):
            state = env.reset()
            sum_reward = 0
            t = 0
            done = False
            
            while not done:
                if e > params['start_at']:
                    action = agent.get_action(state)
                else:
                    action = env.action_space.sample()
                    
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                sum_reward += reward
                t += 1
                
                # for agents using online training
                if agent.online and e > params['start_at']:
                    agent.learn()
                
            # for agents using offline training
            if not agent.online and e > params['start_at']:
                agent.learn()
            
            ep = {'episode':e, 't':t, 'sum_reward':sum_reward, 'optim_steps':agent.optim_steps, 'memory':len(agent.memory)}
            log['episodes'].append(ep)
            if verbose:
                print(ep)    

            if e % save_freq == 0:                
                agent.save(model_dir + params['file_name'] + '.pth')
                with open(log_dir + params['file_name'] + '.pkl', 'wb') as f:
                    pickle.dump(log, f)
                if verbose:
                    print('Episode ' + str(e) + ': Saved model weights and log.')
        env.close()
        
    except Exception:
        traceback.print_exc()
        breakpoint()
        
if __name__ == '__main__':
    args = parser.parse_args()
    env = gym.make(args.env_name).unwrapped
    # train(...) # TODO