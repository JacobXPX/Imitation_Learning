import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import time
from BehaviorCloning import BCPolicy, GraphGenerator

"""
Code to evaluate behavioral cloning policy.
Example usage:
    python bc_eval.py Humanoid-v1 --render --num_rollouts 20
"""

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    return args

def rollout(policy, env, args):
    max_steps = env.spec.timestep_limit
    r_list, o_list, a_list = [], [], []
    total_step_n = []

    for i in range(args.num_rollouts):
        print('Iteration: ', i)
        o = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            a = policy.evaluate(o[None, :])
            o_list.append(o)
            a_list.append(a)
            o, r, done, _ = env.step(a)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 500 == 0: print("%i/%i"%(steps, max_steps))
            time.sleep(0.05)
            if steps >= max_steps:
                break
        r_list.append(totalr)
        total_step_n.append(steps)
    print('steps', total_step_n)
    print('returns', r_list)
    print('mean return', np.mean(r_list))
    print('std of return', np.std(r_list))


def main():
    import gym

    args = get_args()
    env = gym.make(args.envname)

    print('load expert data...')

    with open('experts/' + args.envname + '_expert.pkl', 'rb') as f:
        expert_data = pickle.load(f)

    dim_ao = [expert_data['observations'].shape[1], expert_data['actions'].shape[2]]
    print('obs and action shape: ',dim_ao)

    print('load bc policy...')
    my_graph = GraphGenerator(size=[256, 256, 64], bbeta=0.001, dim_ao = dim_ao, learning_rate = 0.001)
    bc_policy = BCPolicy(bc_graph = my_graph)
    bc_policy.eval('trained_model')
    rollout(bc_policy, env, args)

if __name__ == '__main__':
    main()
