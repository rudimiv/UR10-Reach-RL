from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import FlattenObservation

import torch
import numpy as np

import argparse
import sys

from UR10 import UR10


def test(path, alg='PPO', space='sphere', evals_num=20, complex_env=True, force=0.1, pos_range=0.5, max_steps=500):
    if complex_env:
        angle_control=True
        complex_obs_space=True
    else:
        angle_control=False
        complex_obs_space=False
        
    print(f'RL ALG: {alg} Space: {space} Evals Num: {evals_num} Angle Control: {angle_control}')
    
    env_test = FlattenObservation(UR10(
            is_dense=False, is_train=True, is_fixed=False, angle_control=angle_control, 
            force=force,  complex_obs_space=complex_obs_space, pos_range=pos_range, 
            max_steps=max_steps, space=space
        ))
    
    if alg.upper() == 'PPO':
        model = PPO.load(path)
    else:
        model = DDPG.load(path)
        
    res = evaluate_policy(model, env_test, evals_num, return_episode_rewards=True, deterministic=True)[1]
    
    print('Success Rate:', np.sum(np.array(res) < max_steps) / len(res))
    print('Mean episode length:', np.mean(np.array(res)))
    

def train(alg='PPO', space='sphere', complex_env=True, frames=2000000, max_steps=500, force=0.1, pos_range=0.5, path=None):
    if complex_env:
        angle_control=True
        complex_obs_space=True
    else:
        angle_control=False
        complex_obs_space=False
    
    
    env = FlattenObservation(UR10(
        is_train=True, is_fixed=False, angle_control=angle_control, 
        force=force, complex_obs_space=complex_obs_space, 
        max_steps=max_steps, pos_range=pos_range,
        space=space
    ))
    # Normalize

    eval_env = FlattenObservation(UR10(
        is_dense=False, is_train=True, is_fixed=False, angle_control=angle_control,
        force=force, complex_obs_space=complex_obs_space, 
        max_steps=max_steps, pos_range=pos_range
    ))
    
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=200, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/bestmodels/',
                             n_eval_episodes=20, eval_freq=20000, callback_after_eval=stop_train_callback, 
                             deterministic=True, render=False, verbose=1)

    '''vec_env = SubprocVecEnv(
        [lambda: FlattenObservation(UR10(is_train=True, is_fixed=False, angle_control=True, is_dense=True, force=0.1))] * 6
    )'''

    if alg.upper() == 'PPO':
        policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log='./tesnsorboard_log_2/',
            policy_kwargs=policy_kwargs, learning_rate=0.0003
                    # learning_rate=linear_schedule(0.0003)
                   )
    
    else:
        n_actions = env.action_space.shape[-1]
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions), dtype=np.float64)
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions, dtype=np.float32), sigma=0.005 * np.ones(n_actions, dtype=np.float32))
        model = DDPG("MlpPolicy", env, learning_rate=1e-4, batch_size=32, action_noise=action_noise, tau=0.001, verbose=1, tensorboard_log='./tesnsorboard_log/')

    model.learn(total_timesteps=frames, progress_bar=True, callback=eval_callback)
    
    if path is not None:
        model.save(path)

    

    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('-p', '--path', type=str)
    parser.add_argument('--evals', default=20, type=int)
    
    parser.add_argument("-m", "--model", default='ppo', type=str, help="model")
    parser.add_argument("-s", "--space", default='sphere', type=str, help="space [sphere or cube]")
    parser.add_argument("-c", "--complex", default=True, type=bool, help="Complex task (without IK)")
    parser.add_argument("-l", "--frames", default=1000000, type=int, help="total frames")
    parser.add_argument("-e", "--max_steps", default=500, type=int, help="max steps in episode")
    parser.add_argument("-f", "--force", default=0.1, type=float, help="force of actions")
    parser.add_argument("-r", "--pos_range", default=0.5, type=float, help="pos range for sphere")
    
    args = parser.parse_args(sys.argv[1:])
    
    if args.train:
        train(
            alg=args.model, 
            space=args.space, 
            complex_env=args.complex,
            frames=args.frames,
            max_steps=args.max_steps,
            force=args.force,
            pos_range=args.pos_range
        )
    
    if args.test:
        test(
            path=args.path,
            alg=args.model,
            space=args.space,
            evals_num=args.evals,
            complex_env=args.complex,
            force=args.force,
            pos_range=args.pos_range,
            max_steps=args.max_steps
        )
    
if __name__ == '__main__':
    main()