from environment import Environment
from helper.dicts.convert_actions import *
from helper.constants.settings import *

import gym
from gym import spaces
from gym.envs.registration import register
from copy import deepcopy


class ProfitGym(Environment, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        width,
        height,
        turns,
        products: dict,
        render_mode=None,
        copy=False,
        old_env=None,
    ):
        if copy:
            super().__init__(
                old_env.unwrapped.width,
                old_env.unwrapped.height,
                old_env.unwrapped.turns,
                old_env.unwrapped.products,
                copy=copy,
                grid=old_env.unwrapped.grid,
                buildings=old_env.unwrapped.buildings,
            )
            # print(
            #     f"currentb {type(old_env.unwrapped.current_building)}\ntarget {type(old_env.unwrapped.target_building)}\nbuildings: {type(old_env.buildings)}"
            # )
            self.current_building = deepcopy(old_env.unwrapped.current_building)
            self.target_building = deepcopy(old_env.unwrapped.target_building)
            self.buildings = [
                deepcopy(b)
                for b in old_env.unwrapped.buildings
                if b is not old_env.unwrapped.target_building
            ]
            self.buildings.append(self.target_building)

            self.observation_shape = (
                old_env.unwrapped.height + 2,
                old_env.unwrapped.width + 2,
                NUM_CHANNELS,
            )
            self.observation_space = old_env.unwrapped.observation_space

        else:
            super().__init__(width, height, turns, products)

            # [obstacles, inputs, agent's single output] each in a 100x100 grid
            # channels last for tensorflow
            self.observation_shape = (self.height + 2, self.width + 2, NUM_CHANNELS)
            self.observation_space = spaces.MultiBinary(self.observation_shape)

        # We have 16 different buildings (TODO: +4 for combiners) at four possible positions (at most 3 valid) adjacent to the input tile
        self.action_space = spaces.MultiDiscrete((NUM_SUBBUILDINGS, NUM_DIRECTIONS))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, obstacle_probability=0.1, seed=None, options=None, copy=False):
        if copy:
            return
        super().reset(seed=seed)

        if SIMPLE_GAME:
            distance_range = range(3, self.width + self.height, 2)
        else:
            distance_range = range(6, self.width + self.height)

        start_building, factory = self.task_generator.generate_task(
            obstacle_probability, distance_range
        )
        self.current_building = start_building
        self.target_building = factory

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

        done = self.is_connected(self.current_building, self.target_building)
        reward = SUCCESS_REWARD if done else (LEGAL_REWARD if legal else ILLEGAL_REWARD)
        state = self.grid_to_observation()
        info = {}

        return state, reward, done, legal, info

    @staticmethod
    def split_action(action_id):
        positional_action = action_id // NUM_SUBBUILDINGS
        building_action = action_id % NUM_SUBBUILDINGS

        return positional_action, building_action

    def get_building_from_action(self, action_id):
        positional_action, building_action = self.split_action(action_id)

        x, y = self.current_building.get_output_positions()[0]
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

    def grid_to_observation(self):
        padded_grid = np.pad(self.grid, pad_width=1, constant_values="x")

        # empty = np.where(np.isin(padded_grid, [" ", "<", ">", "^", "v"]), 1.0, 0.0)
        obstacles = np.where(padded_grid != " ", 1.0, 0.0)

        # target_x, target_y = self.target_building.x, self.target_building.y
        target_input_positions = self.target_building.get_input_positions()
        input_idx = target_input_positions[:, 1], target_input_positions[:, 0]
        inputs = np.zeros((self.height + 2, self.width + 2), dtype=np.float32)
        inputs[input_idx] = 1

        agent_x, agent_y = self.current_building.get_output_positions()[0]
        output = np.zeros((self.height + 2, self.width + 2), dtype=np.float32)
        output[(agent_y, agent_x)] = 1

        channels_first = np.stack([obstacles, inputs, output])
        channels_last = np.moveaxis(channels_first, 0, 2)
        return channels_last


def register_gym():
    register(id=GYM_ID, entry_point="profit_gym:ProfitGym")


def make_gym(width, height, copy=False, old_env=None):
    return gym.make(
        GYM_ID,
        width=width,
        height=height,
        turns=50,
        products={},
        render_mode=None,
        copy=copy,
        old_env=old_env,
    )
