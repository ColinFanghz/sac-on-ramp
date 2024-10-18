"""
@Author: Fhz
@Create Date: 2024/1/3 9:17
@File: get_args.py
@Description: 
@Modify Person Date: 
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # SUMO config
    parser.add_argument("--count", type=int, default=0, help="The length of a training episode.")
    parser.add_argument("--show_gui", type=bool, default=False, help="The flag of show SUMO gui.")
    parser.add_argument("--sumocfgfile", type=str, default="../sumo_config/my_config_file.sumocfg", help="The path of the SUMO configure file.")
    parser.add_argument("--egoID", type=str, default="self_car", help="The ID of ego vehicle.")
    parser.add_argument("--start_time", type=int, default=24, help="The simulation step before learning.")
    parser.add_argument("--collision", type=bool, default=False, help="The flag of collision of ego vehicle.")
    parser.add_argument("--test", type=bool, default=False, help="The flag of test.")
    parser.add_argument("--sleep", type=bool, default=False, help="The flag of sleep of each simulation.")
    parser.add_argument("--lane_width", type=float, default=3.2, help="The width of sub lane in SUMO config.")
    parser.add_argument("--y_none", type=float, default=2000.0, help="The longitudinal position of a none exist vehicle.")
    parser.add_argument("--num_action", type=int, default=142, help="The number of action space.")
    parser.add_argument("--lane_change_time", type=float, default=5, help="The time of lane change.")
    parser.add_argument("--prob_main", type=float, default=0.8, help="The probability of main lane.")
    parser.add_argument("--prob_merge", type=float, default=0.1, help="The probability of merge lane.")
    parser.add_argument("--seed_value", type=str, default="123", help="The seed value.")
    parser.add_argument("--seed_value1", type=str, default="456", help="The seed value1.")
    parser.add_argument("--gamma", type=str, default="0.99", help="The reward param.")

    # Road config
    parser.add_argument("--min_x_position", type=float, default=-40.0, help="The minimum lateral position of vehicle.")
    parser.add_argument("--max_x_position", type=float, default=0.0, help="The maximum lateral position of vehicle.")
    parser.add_argument("--min_y_position", type=float, default=-1.0, help="The minimum longitudinal position of vehicle.")
    parser.add_argument("--max_y_position", type=float, default=1500.0, help="The maximum longitudinal position of vehicle.")
    parser.add_argument("--min_x_speed", type=float, default=-3.0, help="The minimum lateral speed of vehicle.")
    parser.add_argument("--max_x_speed", type=float, default=3.0, help="The maximum lateral speed of vehicle.")
    parser.add_argument("--min_y_speed", type=float, default=0.0, help="The minimum longitudinal speed of vehicle.")
    parser.add_argument("--max_y_speed", type=float, default=40.0, help="The maximum longitudinal speed of vehicle.")
    parser.add_argument("--min_x_acc", type=float, default=-4.5, help="The minimum lateral acceleration of vehicle.")
    parser.add_argument("--max_x_acc", type=float, default=2.5, help="The maximum lateral acceleration of vehicle.")
    parser.add_argument("--min_y_acc", type=float, default=-4.5, help="The minimum longitudinal acceleration of vehicle.")
    parser.add_argument("--max_y_acc", type=float, default=2.5, help="The maximum longitudinal acceleration of vehicle.")
    parser.add_argument("--gap", type=float, default=10.0, help="The threshold of ego vehicle to other vehicle.")
    parser.add_argument("--leaderMaxDecel", type=float, default=4.5, help="The leader maximum deceleration.")

    # Reward config
    parser.add_argument("--w_jerk_x", type=float, default=0.1, help="The weight of lateral jerk reward.")
    parser.add_argument("--w_jerk_y", type=float, default=0.1, help="The weight of longitudinal jerk reward.")
    parser.add_argument("--w_time", type=float, default=0.1, help="The weight of time consuming reward.")
    parser.add_argument("--w_lane", type=float, default=2, help="The weight of target lane reward.")
    parser.add_argument("--w_speed", type=float, default=0.1, help="The weight of desired speed reward.")
    parser.add_argument("--R_time", type=float, default=-1, help="The reward of time consuming.")
    parser.add_argument("--P_lane", type=float, default=-8.0, help="The lateral position of target lane.")
    parser.add_argument("--V_desired", type=float, default=30.0, help="The desired speed.")
    parser.add_argument("--R_collision", type=float, default=-400, help="The reward of ego vehicle collision.")

    # Done config
    parser.add_argument("--target_lane_id", type=int, default=1, help="The ID of target lane.")
    parser.add_argument("--merge_position", type=float, default=422.79, help="The position of the merge lane.")
    parser.add_argument("--max_count", type=int, default=60, help="The maximum length of a training episode.")

    # save train data
    parser.add_argument("--train_flag", type=bool, default=False, help="The flag of training.")
    parser.add_argument("--total_reward", type=str, default="totalReward.npy", help="The total reward of training.")
    parser.add_argument("--comfort_reward", type=str, default="comfortReward.npy", help="The comfort reward of training.")
    parser.add_argument("--efficiency_reward", type=str, default="efficiencyReward.npy", help="The efficiency reward of training.")
    parser.add_argument("--safety_reward", type=str, default="safetyReward.npy", help="The safety reward of training.")
    parser.add_argument("--total_loss", type=str, default="totalLoss.npy", help="The total loss of training.")
    parser.add_argument("--success_rate", type=str, default="successRate.npy", help="The success rate of training.")
    parser.add_argument("--collision_rate", type=str, default="collisionRate.npy", help="The collision rate of training.")

    args = parser.parse_args()

    return args

