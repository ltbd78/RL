import pickle
import argparse
import traceback
import os

from RL.agents import *


AGENTS = {
    'random': RandomAgent,
    'dqn': DQNAgent,
    'a2c': A2CAgent,
    'td3': TD3Agent,
}

def train(agent_type, env, verbose=True, save_freq=50, save_dir='./', **params):
    if verbose:
        print(params)

    agent = AGENTS[agent_type](env.observation_space, env.action_space, **params)

    log = {'agent':agent_type, 'params':params, 'episodes':[]}

    if save_dir[-1] != '/':
        raise NotADirectory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        ep = 0
        t_total = 0
        while t_total < params['max_steps']:
            state, info = env.reset()
            sum_reward = 0
            t_ep = 0
            done = False

            while not done:
                if t_total > params['start_at']:
                    action = agent.get_action(state)
                else:
                    action = env.action_space.sample()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                sum_reward += reward
                t_ep += 1

                # for agents using online training
                if agent.online and t_total > params['start_at']:
                    agent.learn()

            # for agents using offline training
            if not agent.online and t_total > params['start_at']:
                agent.learn()

            ep += 1
            t_total += t_ep
            ep_info = {'episode':ep, 't_ep':t_ep, 't_total':t_total, 'sum_reward':sum_reward, 'optim_steps':agent.optim_steps, 'memory':len(agent.memory)}
            log['episodes'].append(ep_info)
            if verbose:
                print(ep_info)

            if ep % save_freq == 0:
                agent.save(save_dir + params['file_name'] + '.pth')
                with open(save_dir + params['file_name'] + '.pkl', 'wb') as f:
                    pickle.dump(log, f)
                if verbose:
                    print('Episode ' + str(ep) + ': Saved model weights and log.')
        env.close()

    except Exception:
        traceback.print_exc()
        breakpoint()
