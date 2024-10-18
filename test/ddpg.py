"""
@Author: Fhz
@Create Date: 2023/6/2 9:17
@File: ddpg.py
@Description: 
@Modify Person Date: 
"""
import sys
sys.path.append('../env')
sys.path.append('../args')

from stable_baselines3 import DDPG
from env_ss1 import *
from get_args import *


def get_returns(arr, gamma):

    returns = 0

    for i in range(len(arr)):
        returns = returns + arr[i] * gamma ** i

    return returns


if __name__ == '__main__':
    args = get_args()
    args.test = True
    env = SumoGym(args)
    
    models = ["../models/model_ddpg1", "../models/model_ddpg2", "../models/model_ddpg3"]

    for model_tmp in models:
        model = DDPG.load(model_tmp, env=env)

        gamma = 0.99
        eposides = 200
        rewards = []
        speeds = []
        success_count = 0
        collision_count = 0
        counts = 0

        for eq in range(eposides):
            print("Test eposide: {}".format(eq))
            obs = env.reset()
            env.seed_value = "123{}".format(eq)
            env.seed_value1 = "456{}".format(eq)

            done = False

            count = 0
            reward_tmp = []
            speed_tmp = 0
            while not done:
                count += 1
                time.sleep(0.01)
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                reward_tmp.append(reward)
                speed_tmp = speed_tmp + info["speeds"]

            reward_tmp_ave = get_returns(reward_tmp, gamma)
            speed_tmp_ave = speed_tmp / count

            rewards.append(reward_tmp_ave)
            speeds.append(speed_tmp_ave)
            print(info)

            counts = counts + count
            if info["success"]:
                success_count = success_count + 1

            if info["collision"]:
                collision_count = collision_count + 1

        rewards = np.array(rewards)
        speeds = np.array(speeds)

        print(
            "The rewards is: {}, the robustness is: {}, the speed is: {},  the success_rate is: {},  the collision_rate is: {}, the counts is : {}".format(
                rewards.mean(), rewards.std(), speeds.mean(), success_count / eposides, collision_count / eposides,
                                                           counts / eposides))



