from environment import Environment
from helper.dicts.convert_actions import *
from helper.constants.settings import *

import gym
from gym import spaces


class ProfitGym(Environment, gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, width, height, turns, products: dict, render_mode=None):
        super().__init__(width, height, turns, products)

        # [obstacles, inputs, agent's single output] each in a 100x100 grid
        self.observation_space = spaces.MultiBinary((3, MAX_HEIGHT, MAX_WIDTH))

        # We have 16 different buildings (TODO: +4 for combiners) at four possible positions (at most 3 valid) adjacent to the input tile
        self.action_space = spaces.MultiDiscrete((16, 4))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # task generator modifies self (this environment!)
        mine, factory = self.task_generator.generate_simple_task(seed)
        self.current_building = mine
        self.target_building = factory

        observation = self.grid_to_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        building_action, positional_action = action

        x, y = self.current_building.get_output_positions()[0]
        x_offset, y_offset = POSITIONAL_ACTION_TO_DIRECTION[positional_action]
        input_x, input_y = x + x_offset, y + y_offset

        BuildingClass, subtype = BUILDING_ACTION_TO_CLASS_SUBTYPE[building_action]
        new_building = BuildingClass.from_input_position(input_x, input_y, subtype)

        terminated = False
        truncated = True

        legal_action = self.would_connect_to(self.current_building, new_building)
        if self.is_legal_position(new_building) and legal_action:
            self.add_building(new_building)

            if not self.has_connection_loop(self.current_building, new_building):
                truncated = False
                terminated = self.is_connected(new_building, self.target_building)

        # sparse rewards for now
        reward = 1 if terminated else (-1 if truncated else 0)

        observation = self.grid_to_observation()
        info = {}
        if self.render_mode == "human":
            self.render()

        self.current_building = new_building

        return observation, reward, terminated, truncated, info

    def render(self):
        print(self)

    def grid_to_observation(self):
        obstacles = np.where(self.grid != " ", True, False).astype(bool)

        target_input_positions = self.target_building.get_input_positions()
        inputs = np.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=bool)
        inputs[np.array(target_input_positions)] = True

        agent_output = self.current_building.get_output_positions()[0]
        output = np.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=bool)
        output[agent_output] = True

        return np.stack([obstacles, inputs, output])
