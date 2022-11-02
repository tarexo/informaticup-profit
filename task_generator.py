from environment import Environment
from classes.buildings import *


def place_mine(env: Environment, deposit: Deposit, factory: Factory):
    min_distance = None
    best_mine = None
    for candidate_input_position in deposit.get_adjacent_output_positions():
        for subtype in range(4):
            # candidate_position = where to place mine of certain subtype so that its input is adjacent to the deposits output
            candidate_position = (
                candidate_input_position - shape.input_position + shape.center_position
            )
            mine = Mine(candidate_position, subtype)
            if env.is_legal_position(mine):
                mine_output_position = candidate_position - shape.output_position
                distance = heuristic(mine_output_position, factory)
                if min_distance is None or distance <= min_distance:
                    min_distance = distance
                    best_mine = mine

    assert best_mine
    env.add_building(best_mine)


def connect_deposit_factory(env, deposit, factory):
    place_mine(env, deposit, factory)


if __name__ == "__main__":
    env = Environment(
        100,
        100,
        100,
        [
            {
                "type": "product",
                "subtype": 0,
                "resources": [1, 0, 0, 0, 0, 0, 0, 0],
                "points": 10,
            }
        ],
    )

    deposit = Deposit((2, 2), 3, 3, 0)
    factory = Factory((env.width - 7, env.height - 7), 0)

    env.add_building(deposit)
    env.add_building(factory)

    connect_deposit_factory(env, deposit, factory)
