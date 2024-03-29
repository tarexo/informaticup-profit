from environment.environment import Environment
from helper.convert_actions import *
from settings import *

import numpy as np
import gym
from gym import spaces

import random


class ProfitGym(Environment, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, width, height, field_of_vision, turns, products, time=120):
        super().__init__(width, height, turns, products, time=time)

        self.field_of_vision = field_of_vision

        self.vision_shape = (self.field_of_vision, self.field_of_vision, 3)
        self.legal_action_shape = (NUM_ACTIONS,)
        self.target_dir_shape = (6,)
        self.observation_space = spaces.Tuple(
            [
                spaces.MultiBinary(self.vision_shape),
                spaces.MultiBinary(self.legal_action_shape),
                spaces.MultiBinary(self.target_dir_shape),
            ],
        )

        # We have 16 different buildings (TODO: +4 for combiners) at four possible positions (at most 3 valid) adjacent to the input tile
        self.action_space = spaces.MultiDiscrete((NUM_SUBBUILDINGS, NUM_DIRECTIONS))

    def set_task(self, start_building, factory):
        self.current_building = start_building
        self.target_building = factory

        self.current_path = [start_building]

        num_outlets = len(self.current_building.get_output_positions())
        self.outlet = random.randrange(num_outlets)

        self.target_distance = self.get_min_distance(
            self.current_building, self.target_building
        )

    def reset(self, difficulty=0.0, seed=None, options=None):
        super().reset(seed=seed)

        start_building, factory = self.task_generator.generate_task(difficulty)
        self.set_task(start_building, factory)

        state = self.grid_to_observation()
        info = {}

        return state, info

    def step(self, action_tensor):
        new_building = self.get_building_from_action(int(action_tensor))

        legal = False
        if self.is_legal_action(new_building):
            legal = True
            self.add_building(new_building)
            self.current_building = new_building
            self.current_path.append(new_building)
            num_outlets = len(self.current_building.get_output_positions())
            self.outlet = random.randrange(num_outlets)
            self.make_untargetable([new_building])

        done = self.is_connected(self.current_building, self.target_building)

        old_target_distance = self.target_distance
        self.target_distance = self.get_min_distance(
            self.current_building, self.target_building
        )
        distance_reduction = old_target_distance - self.target_distance

        reward = self.calculate_reward(done, legal, distance_reduction)

        state = self.grid_to_observation()
        info = {}

        return state, reward, done, legal, info

    @staticmethod
    def calculate_reward(done, legal, distance_reduction):
        if done:
            return SUCCESS_REWARD
        elif not legal:
            return ILLEGAL_REWARD
        return LEGAL_REWARD + (DISTANCE_REDUCTION_REWARD * distance_reduction)

    @staticmethod
    def split_action(action_id):
        positional_action = action_id // NUM_SUBBUILDINGS
        building_action = action_id % NUM_SUBBUILDINGS

        return positional_action, building_action

    def get_building_from_action(self, action_id):
        positional_action, building_action = self.split_action(action_id)

        x, y = self.current_building.get_output_positions()[self.outlet]
        x_offset, y_offset = POSITIONAL_ACTION_TO_DIRECTION[positional_action]
        input_x, input_y = x + x_offset, y + y_offset

        BuildingClass, subtype = BUILDING_ACTION_TO_CLASS_SUBTYPE[building_action]
        new_building = BuildingClass.from_input_position(input_x, input_y, subtype)

        return new_building

    def is_legal_action(self, new_building):
        legal_connection = self.would_connect_to(self.current_building, new_building)
        return legal_connection and self.is_legal_position(new_building)

    def render(self):
        print(self)

    def get_field_of_vison(self):
        padding = self.field_of_vision // 2
        padded_grid = np.pad(self.grid, pad_width=padding, constant_values="x")

        agent_x, agent_y = self.current_building.get_output_positions()[self.outlet]
        end_x = agent_x + self.field_of_vision
        end_y = agent_y + self.field_of_vision

        grid_fov = padded_grid[agent_y:end_y, agent_x:end_x]
        grid_fov[grid_fov == "<"] = "~"
        grid_fov[grid_fov == ">"] = "~"
        grid_fov[grid_fov == "^"] = "~"
        grid_fov[grid_fov == "v"] = "~"

        obstacles_fov = np.where(grid_fov != " ", 1.0, 0.0)
        target_fov = np.where(grid_fov == "+", 1.0, 0.0)
        tunnels_fov = np.where(grid_fov == "~", 1.0, 0.0)

        for building in self.current_path:
            false_targets = building.get_input_positions()
            idx = false_targets - (agent_x - padding, agent_y - padding)
            for x, y in idx:
                if x > 0 and y > 0:
                    if x < self.field_of_vision and y < self.field_of_vision:
                        target_fov[y, x] = 0.0

        return np.stack([obstacles_fov, target_fov, tunnels_fov], axis=2).astype(
            np.int8
        )

    def get_legal_actions(self):
        legal_actions = np.zeros(self.legal_action_shape, dtype=np.int8)
        for action in range(NUM_ACTIONS):
            new_building = self.get_building_from_action(action)
            if self.is_legal_position(new_building):
                legal_actions[action] = 1

        return legal_actions

    def get_target_direction(self):
        agent_x, agent_y = self.current_building.get_output_positions()[self.outlet]
        target_x, target_y = self.target_building.get_center_position()

        x_distance = agent_x - target_x
        y_distance = agent_y - target_y

        target_position = np.zeros(self.target_dir_shape, dtype=np.int8)
        if x_distance > 2:
            target_position[0] = 1
        elif x_distance < 2:
            target_position[1] = 1
        else:
            target_position[2] = 1

        if y_distance > 2:
            target_position[3] = 1
        elif y_distance < 2:
            target_position[4] = 1
        else:
            target_position[5] = 1

        return target_position

    def grid_to_observation(self):
        return (
            self.get_field_of_vison(),
            self.get_legal_actions(),
            self.get_target_direction(),
        )
