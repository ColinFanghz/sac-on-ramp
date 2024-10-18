"""
@Author: Fhz
@Create Date: 2023/6/2 9:17
@File: Merge_ppo.py
@Description: 
@Modify Person Date: 
"""
import sys
sys.path.append('../env')
sys.path.append('../args')

from stable_baselines3 import TD3
from env_ss1 import *
from get_args import *


if __name__ == '__main__':
    args = get_args()
    env = SumoGym(args)

    model = TD3('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[64, 64]),
                learning_rate=5e-4,
                batch_size=256,
                gamma=0.99,
                verbose=1,
                tensorboard_log="logs/",
                )

    model.learn(int(1e5), log_interval=1)
    model.save("../models/model_td3")

