import gym
import numpy as np

env = gym.make('FrozenLake-v0')
alpha = 0.4
gamma = 0.999
q_table = dict([(x, [1, 1, 1, 1]) for x in range(16)])


def choose_action(o):
    return np.argmax(q_table[o])


for i in range(10000):
    obs = env.reset()
    act = choose_action(obs)

    prev_obs = None
    prev_act = None

    t = 0

    for t in range(2500):
        env.render()
        obs, reward, done, info = env.step(act)
        act = choose_action(obs)
        if prev_obs is not None:
            q_old = q_table[prev_obs][prev_act]
            q_new = q_old
            if done:
                q_new += alpha * (reward - q_old)
            else:
                q_new += alpha * (reward + gamma * q_table[obs][act] - q_old)

            new_table = q_table[prev_obs]
            new_table[prev_act] = q_new

            q_table[prev_obs] = new_table

        prev_obs = obs
        prev_act = act

        if done:
            print("Episode {} finished after {} steps with r={}.".format(i, t, reward))
            break
