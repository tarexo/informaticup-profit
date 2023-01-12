from settings import *
from model.architecture import ActorCritic, DeepQNetwork
from environment.setup import register_gym, make_gym
from helper.convert_actions import action_to_description

import statistics
import os


def evaluate(env, model, difficulty, num_episodes, force_legal=False):
    rewards = []
    for _episode in range(num_episodes):
        state, _ = env.reset(difficulty=difficulty)
        _, episode_reward = model.run_episode(
            state, exploration_rate=0, greedy=True, force_legal=force_legal
        )

        rewards.append(episode_reward)
    return statistics.mean(rewards)


def evaluate_model(env, model, num_episodes=100):
    print(f"Testing model for {num_episodes} episodes...")
    validation_score = evaluate(
        env, model, difficulty=1.0, num_episodes=num_episodes, force_legal=True
    )
    print(f"Model achieved a Test score of {validation_score}\n")

    return validation_score


def evaluate_models(width, height, model_paths):
    best_score = 0.0
    best_model = None
    best_model_name = None

    print(f"\nComparing all models on a {width}x{height} grid...\n")
    for model_path in model_paths:
        model_name = model_path.split("\\")[-1]
        game_type, field_of_vision, network = model_name.split("__")
        field_of_vision = int(field_of_vision.split("x")[0])

        env = make_gym(width, height, field_of_vision)

        if "DQN" in network:
            model = DeepQNetwork(env)
        elif "A-C" in network:
            model = ActorCritic(env)
        model.load(model_path)

        validation_score = evaluate_model(env, model)
        if validation_score > best_score:
            best_score = validation_score
            best_model = model
            best_model_name = model_name

    print(f"Best Model: {best_model_name}")
    best_model.summary()


def compare_all_saved_model(width, height):
    models_path = os.path.join(".", "saved_models")
    model_names = os.listdir(models_path)
    model_paths = [os.path.join(models_path, model_name) for model_name in model_names]
    evaluate_models(width, height, model_paths)


def check_model_sanity(env, model, difficulty):
    state, _ = env.reset(difficulty=difficulty)

    for _ in range(MAX_STEPS_EACH_EPISODE):
        if DEBUG:
            print(f"\nField of Vision:")
            print(state[0][:, :, 1])

        greedy_action = model.verbose_greedy_prediction(state)
        state, reward, done, legal, _info = env.step(greedy_action)

        direction_id, subbuilding_id = env.split_action(greedy_action)
        action_description = action_to_description(direction_id, subbuilding_id)

        print(
            f"\nGreedy Action: {action_description}"
            + (" (illegal)" if not legal else "")
        )
        print("--> Reward:", reward)
        if done or not legal:
            break

    env.render()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    register_gym()

    for size in [20, 30, 50]:
        compare_all_saved_model(size, size)
