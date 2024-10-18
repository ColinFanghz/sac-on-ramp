"""
@Author: Fhz
@Create Date: 2024/1/3 9:17
@File: env_ss1.py
@Description: 
@Modify Person Date: 
"""
import os
import random
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import gym
from gym import spaces
import numpy as np
import math
import traci
from sumolib import checkBinary
import time


class SumoGym(gym.Env):
    def __init__(self, args):
        self.current_state = None
        self.Acc = None
        self.heading = None
        self.vehicles = None
        self.prob_main = args.prob_main
        self.prob_merge = args.prob_merge
        self.seed_value = args.seed_value
        self.seed_value1 = args.seed_value1
        self.test = args.test

        # Record log
        self.is_success = False
        self.total_reward = 0
        self.comfort_reward = 0
        self.efficiency_reward = 0
        self.safety_reward = 0
        self.done_count = 0

        # SUMO config
        self.count = args.count
        self.show_gui = args.show_gui
        self.sumocfgfile = args.sumocfgfile
        self.egoID = args.egoID
        self.start_time = args.start_time
        self.collision = args.collision
        self.sleep = args.sleep
        self.lane_width = args.lane_width
        self.y_none = args.y_none
        self.num_action = args.num_action
        self.lane_change_time = args.lane_change_time
        self.leaderMaxDecel = args.leaderMaxDecel

        # Road config
        self.min_x_position = args.min_x_position
        self.max_x_position = args.max_x_position
        self.min_y_position = args.min_y_position
        self.max_y_position = args.max_y_position
        self.min_x_speed = args.min_x_speed
        self.max_x_speed = args.max_x_speed
        self.min_y_speed = args.min_y_speed
        self.max_y_speed = args.max_y_speed
        self.min_x_acc = args.min_x_acc
        self.max_x_acc = args.max_x_acc
        self.min_y_acc = args.min_y_acc
        self.max_y_acc = args.max_y_acc
        self.gap = args.gap

        # Reward config
        self.w_jerk_x = args.w_jerk_x
        self.w_jerk_y = args.w_jerk_y
        self.w_time = args.w_time
        self.w_lane = args.w_lane
        self.w_speed = args.w_speed
        self.R_time = args.R_time
        self.P_lane = args.P_lane
        self.V_desired = args.V_desired
        self.R_collision = args.R_collision

        # Done config
        self.target_lane_id = args.target_lane_id
        self.merge_position = args.merge_position
        self.max_count = args.max_count

        self.low = np.array([
            # ego vehicle
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            # ego_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            # ego_follower
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            # ego_left_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            # ego_left_follower
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
            # ego_right_leader
            self.min_x_position,
            self.min_y_position,
            self.min_x_speed,
            self.min_y_speed,
            self.min_x_acc,
            self.min_y_acc,
        ], dtype=np.float32)

        self.high = np.array([
            # ego vehicle
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            # ego_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            # ego_follower
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            # ego_left_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            # ego_left_follower
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
            # ego_right_leader
            self.max_x_position,
            self.max_y_position,
            self.max_x_speed,
            self.max_y_speed,
            self.max_x_acc,
            self.max_y_acc,
        ], dtype=np.float32)

  
        self.action_space = spaces.Box(np.array([-4.5, 0.5], dtype=np.float32) , np.array([2.5, 1.5], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        action_long_new = action[0]
        action_lat = int(action[1])

        if self.is_success:
            self.is_success = False

        if self.collision:
            self.collision = False

    
        if self.test:
            random.seed(self.seed_value1)

        for veh in self.vehicles:
            traci.vehicle.setSpeedMode(veh, 0)
            traci.vehicle.setLaneChangeMode(veh, 0)

        traci.simulationStep(self.count)
        if random.random() < self.prob_main:
            traci.vehicle.add(vehID="flow_1_{}".format(self.count), routeID="route1", typeID="typedist1",
                              depart="{}".format(self.count),departLane="0", arrivalLane="{}".format(random.randint(0, 2)))
            traci.vehicle.add(vehID="flow_2_{}".format(self.count), routeID="route1", typeID="typedist1",
                              depart="{}".format(self.count), departLane="1", arrivalLane="{}".format(random.randint(0, 2)))
            traci.vehicle.add(vehID="flow_3_{}".format(self.count), routeID="route1", typeID="typedist1",
                              depart="{}".format(self.count), departLane="2", arrivalLane="{}".format(random.randint(0, 2)))

        if random.random() < self.prob_merge:
            traci.vehicle.add(vehID="flow_4_{}".format(self.count), routeID="route2", typeID="typedist1", depart="{}".format(self.count), departLane="0")

        self.count = self.count + 1

        ego_lane = traci.vehicle.getLaneIndex(self.egoID)

        ego_edge = traci.vehicle.getLaneID(self.egoID)

        if ego_edge[:5] == "gneE3" or ego_edge[:5] == ":gneJ":
            action_lat = 0

        traci.vehicle.setSpeed(self.egoID, max(0, traci.vehicle.getSpeed(self.egoID) + action_long_new))

        speeds = max(0, traci.vehicle.getSpeed(self.egoID) + action_long_new)

        if action_lat:
            if ego_lane >= 0 and (ego_lane + action_lat) <= 2:
                traci.vehicle.changeLane("self_car", "{}".format(ego_lane + action_lat), self.lane_change_time)

        self.current_state = self.getVehicleStates()
        Collision_Nums = traci.simulation.getCollidingVehiclesNumber()

        if Collision_Nums:
            print("collision num:{}".format(Collision_Nums))
            self.collision = True

        done = self.getDoneState(ego_lane, self.current_state[1])
        reward = self.getRewards(action_lat, ego_lane)
        info = {
                "success": self.is_success,
                "total_reward": self.total_reward,
                "comfort_reward": self.comfort_reward,
                "efficiency_reward": self.efficiency_reward,
                "safety_reward": self.safety_reward,
                "collision": self.collision,
                "total_counts": self.count,
                "speeds": speeds,
                }

        if done:
            traci.close()

        return self.current_state, reward, done, info


    def render(self):
        self.show_gui = True
        pass


    def reset(self):

        self.collision = False
        if self.show_gui:
            sumoBinary = checkBinary('sumo-gui')
        else:
            sumoBinary = checkBinary('sumo')

        traci.start([sumoBinary, "-c", self.sumocfgfile])

        if self.sleep:
            time.sleep(2)
        
        if self.test:
            random.seed(self.seed_value)

        lane = 0

        for step in range(self.start_time):
            traci.simulationStep(step)
            self.count = step
            if random.random() < self.prob_main:
                traci.vehicle.add(vehID="flow_1_{}".format(step), routeID="route1", typeID="typedist1",
                                  depart="{}".format(step), departLane="0", arrivalLane="{}".format(random.randint(0, 2)))
                traci.vehicle.add(vehID="flow_2_{}".format(step), routeID="route1", typeID="typedist1",
                                  depart="{}".format(step),  departLane="1", arrivalLane="{}".format(random.randint(0, 2)))
                traci.vehicle.add(vehID="flow_3_{}".format(step), routeID="route1", typeID="typedist1",
                                  depart="{}".format(step), departLane="2", arrivalLane="{}".format(random.randint(0, 2)))

            if random.random() < self.prob_merge:
                traci.vehicle.add(vehID="flow_4_{}".format(step), routeID="route2", typeID="typedist1",
                                  depart="{}".format(step), departLane="0")
            if step == 15:
                traci.vehicle.add(vehID="self_car", routeID="route2", typeID="carA", depart="{}".format(step),
                                  departLane="0")

        self.count = self.count + 1
        self.Acc = traci.vehicle.getAcceleration(self.egoID)
        self.heading = traci.vehicle.getAngle(self.egoID)
        traci.vehicle.setSpeedMode(self.egoID, 0)
        traci.vehicle.setLaneChangeMode(self.egoID, 0)

        return self.getVehicleStates()


    def getRewards(self, action_lat, ego_lane):
        """
        action: action of step
        function: get the reward after action.
        """

        # Comfort reward
        Acc_new = traci.vehicle.getAcceleration(self.egoID)
        heading_new = traci.vehicle.getAngle(self.egoID)

        Acc_new_x = Acc_new * math.cos(math.pi * heading_new / 180)
        Acc_new_y = Acc_new * math.sin(math.pi * heading_new / 180)

        Acc_x = self.Acc * math.cos(math.pi * self.heading / 180)
        Acc_y = self.Acc * math.sin(math.pi * self.heading / 180)

        jerk_x = Acc_new_x - Acc_x
        jerk_y = Acc_new_y - Acc_y

        self.Acc = Acc_new
        self.heading = heading_new

        R_comfort = - self.w_jerk_x * abs(jerk_x) - self.w_jerk_y * abs(jerk_y)

        # Efficient reward
        R_time = - self.R_time
        R_speed = -abs(self.current_state[3] - self.V_desired)

        if self.current_state[0] == self.P_lane:
            R_tar = 300
        else:
            R_tar = 0

        R_eff = self.w_time * R_time + self.w_speed * R_speed

        # Safety Reward
        if self.collision:
            R_col = self.R_collision
        else:
            R_col = 0

        if ego_lane == 0 and action_lat == 0:
            min_dist = abs(self.current_state[7])
        elif ego_lane == 0 and action_lat == 1:
            min_dist = min(abs(self.current_state[7]), abs(self.current_state[13]))
        elif ego_lane == 1 and action_lat == 0:
            min_dist = abs(self.current_state[13])
        else:
            min_dist = min(abs(self.current_state[13]), abs(self.current_state[25]))

        R_TTC = -1 / (min_dist + 0.01)

        R_safe = R_col + R_TTC

        R_comfort = max(-100, R_comfort)
        R_eff = max(-30, R_eff)
        R_safe = max(-500, R_safe)

        Rewards = R_comfort + R_eff + R_safe + R_tar

        self.comfort_reward = R_comfort
        self.efficiency_reward = R_eff
        self.safety_reward = R_safe
        self.total_reward = Rewards

        return Rewards

    def getDoneState(self, ego_lane, ego_y):
        """
        function: get the done state of simulation.
        """
        done = False

        if self.collision:
            done = True
            self.is_success = False
            print("Collision occurs!")
            return done

        if self.current_state[0] == self.P_lane:
            self.done_count += 1
            if self.done_count >= 2:
                done = True
                print("Success merge in!")
                self.is_success = True
            return done

        if self.count >= self.max_count:
            done = True
            print("Over time!")
            return done

        if self.current_state[1] >= self.merge_position:
            done = True
            print("Exceed the max position")
            return done

        return done


    def getVehicleStates(self):
        """
        function: Get all the states of vehicles, observation space.
        """
        self.vehicles = traci.vehicle.getIDList()
        states = self.getStates(self.egoID, self.vehicles)
        egoStates = states[0]
        zaFront = self.getZaFront(states)
        targetFront = self.getTargetFront(states)
        targetRear = self.getTargetRear(states)
        leftFront = self.getLeftFront(states)
        leftRear = self.getLeftRear(states)

        self.current_state = [
            # ego
            egoStates[0],
            egoStates[1],
            egoStates[2],
            egoStates[3],
            egoStates[4],
            egoStates[5],

            # za Front
            zaFront[0] - egoStates[0],
            zaFront[1] - egoStates[1],
            zaFront[2],
            zaFront[3],
            zaFront[4],
            zaFront[5],

            # target Front
            targetFront[0] - egoStates[0],
            targetFront[1] - egoStates[1],
            targetFront[2],
            targetFront[3],
            targetFront[4],
            targetFront[5],

            # target Rear
            egoStates[0] - targetRear[0],
            egoStates[1] - targetRear[1],
            targetRear[2],
            targetRear[3],
            targetRear[4],
            targetRear[5],

            # left Front
            leftFront[0] - egoStates[0],
            leftFront[1] - egoStates[1],
            leftFront[2],
            leftFront[3],
            leftFront[4],
            leftFront[5],

            # left Rear
            egoStates[0] - leftRear[0],
            egoStates[1] - leftRear[1],
            leftRear[2],
            leftRear[3],
            leftRear[4],
            leftRear[5],
        ]

        return self.current_state


    def getVehicleStateViaId(self, vehicle_ID):
        """
        vehicle_ID: vehicle ID
        function: Get the Vehicle's state via vehicle ID
        """

        # Get the state of ego vehicle
        curr_pos = traci.vehicle.getPosition(vehicle_ID)
        y_ego, x_ego = curr_pos[0], curr_pos[1]
        y_ego_speed = traci.vehicle.getSpeed(vehicle_ID)
        x_ego_speed = traci.vehicle.getLateralSpeed(vehicle_ID)
        acc_ego = traci.vehicle.getAcceleration(vehicle_ID)
        yaw = traci.vehicle.getAngle(vehicle_ID)
        x_ego_acc = acc_ego * math.cos(math.pi * yaw / 180)
        y_ego_acc = acc_ego * math.sin(math.pi * yaw / 180)

        return x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc


    def getStates(self, egoID, vehicles):
        """
        egoID: ego ID
        vehicles: all vehicle lists
        function: Get the state of all the vehicles around ego vehicle
        """
        arrays = np.zeros(shape=[len(vehicles), 6])

        x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc = self.getVehicleStateViaId(egoID)
        arrays = np.array([x_ego, y_ego, x_ego_speed, y_ego_speed, x_ego_acc, y_ego_acc])

        for i in range(len(vehicles)):
            if vehicles[i] != egoID:
                x_tmp, y_tmp, x_tmp_speed, y_tmp_speed, x_tmp_acc, y_tmp_acc = self.getVehicleStateViaId(vehicles[i])
                array_tmp = np.array([x_tmp, y_tmp, x_tmp_speed, y_tmp_speed, x_tmp_acc, y_tmp_acc])
                arrays = np.vstack([arrays, array_tmp])

        if len(vehicles) == 1:
            arrays = arrays.reshape(1,6)

        return arrays

    def getZaFront(self, states):
        """
        states: ego vehicle and its surrounding vehicle states
        function: Get the state of za front vehicle. SV1
        """

        ego_leader = traci.vehicle.getLeader(self.egoID)

        if ego_leader is None:
            ego = states[0, :]

            x_ind = ego[1] < states[:, 1]
            y_ind = (np.abs(-11.2 - states[:, 0])) < 3.1
            ind = x_ind & y_ind

            if ind.sum() > 0:
                state_ind = states[ind, :]
                zaFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
            else:
                zaFront = np.asarray([-11.2, 1000, ego[2], ego[3], ego[4], ego[5]])
        else:
            leader_id = ego_leader[0]
            x_leader, y_leader, x_leader_speed, y_leader_speed, x_leader_acc, y_leader_acc = self.getVehicleStateViaId(leader_id)
            zaFront = np.asarray([x_leader, y_leader, x_leader_speed, y_leader_speed, x_leader_acc, y_leader_acc])

        return zaFront

    def getTargetFront(self, states):
        """
        states: ego vehicle and its surrounding vehicle states
        function: Get the state of target front vehicle. SV3
        """
        ego = states[0, :]

        x_ind = ego[1] < states[:, 1]
        y_ind = (np.abs(-8.0 - states[:, 0])) < 3.1
        ind = x_ind & y_ind

        if ind.sum() > 0:
            state_ind = states[ind, :]
            targetFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
        else:
            targetFront = np.asarray([-8.0, 1000, ego[2], ego[3], ego[4], ego[5]])

        return targetFront

    def getTargetRear(self, states):
        """
        states: ego vehicle and its surrounding vehicle states
        function: Get the state of target rear vehicle. SV2
        """
        ego = states[0, :]

        x_ind = ego[1] >= states[:, 1]
        y_ind = (np.abs(-8.0 - states[:, 0])) < 3.1
        ind = x_ind & y_ind

        if ind.sum() > 0:
            state_ind = states[ind, :]
            targetRear = state_ind[(state_ind[:, 1] - ego[1]).argmax(), :]
        else:
            targetRear = np.asarray([-8.0, 1000, ego[2], ego[3], ego[4], ego[5]])

        return targetRear

    def getLeftFront(self, states):
        """
        states: ego vehicle and its surrounding vehicle states
        function: Get the state of left front vehicle. SV5
        """
        ego = states[0, :]

        x_ind = ego[1] < states[:, 1]
        y_ind = (np.abs(-4.8 - states[:, 0])) < 3.1
        ind = x_ind & y_ind

        if ind.sum() > 0:
            state_ind = states[ind, :]
            leftFront = state_ind[(state_ind[:, 1] - ego[1]).argmin(), :]
        else:
            leftFront = np.asarray([-4.8, 1000, ego[2], ego[3], ego[4], ego[5]])

        return leftFront

    def getLeftRear(self, states):
        """
        states: ego vehicle and its surrounding vehicle states
        function: Get the state of left rear vehicle. SV4
        """
        ego = states[0, :]

        x_ind = ego[1] >= states[:, 1]
        y_ind = (np.abs(-4.8 - states[:, 0])) < 3.1
        ind = x_ind & y_ind

        if ind.sum() > 0:
            state_ind = states[ind, :]
            leftRear = state_ind[(state_ind[:, 1] - ego[1]).argmax(), :]
        else:
            leftRear = np.asarray([-4.8, 1000, ego[2], ego[3], ego[4], ego[5]])

        return leftRear
       
       