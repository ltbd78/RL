import gym

from mars.agents.utils.replay_buffer import ReplayBuffer as rb_org
from mars.agents.utils.deprecated.replay_buffer import ReplayBuffer as rb_dep

cartpole_env = gym.make('CartPole-v0').unwrapped
pendulum_env = gym.make('Pendulum-v0').unwrapped

size = 100
state_dim = pendulum_env.observation_space.shape[0]
action_dim = pendulum_env.action_space.shape[0]

rb1 = rb_org(size)
rb2 = rb_dep(state_dim, action_dim, size)

def test_add(rb, env):
    for i in range(1000):
        state = env.reset()
        for j in range(100):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            rb.add(state, action, reward, next_state, done)
            state = next_state

def test_sample(rb):
    for k in range(1000):
        rb.sample(64)


%time test_add(rb1, pendulum_env)
%time test_add(rb2, pendulum_env)

%time test_sample(rb1)
%time test_sample(bb2)