GYM_ID = "Profit"
PRINT_ROUND_LOG = False
DEBUG = True

# Replace Deposit/Factory with single cell variant + only 4 placable buildings: conveyors are 2x1
SIMPLE_GAME = False
# see how envionment performs without any obstacles
NO_OBSTACLES = False

# Model
# which trained model shall be used for the game_solver agent?
GAME_SOLVER_MODEL_NAME = "NORMAL__15x15__DQN_256_128"

# For Training
MODEL_ID = "DQN"  # "DQN" or "A-C"
# hidden layer unit count
NUM_FOV_FEATURES = 256
NUM_COMBINED_FEATURES = 128

# Hyperparameters
MAX_EPISODES = 1000
FINE_TUNE_EPISODES = MAX_EPISODES // 10
MAX_STEPS_EACH_EPISODE = 300
MAX_OBSTACLE_PROBABILITY = 0.3

# what is the minimum reward before increasing task generator difficulty?
# (difficulty ^= distance to factory + number of obstacles)
INCREASE_DIFFICULTY_AT = 0.0
# at what reward level should the difficulty be at its maximum?
MAX_DIFFICULTY_AT = 0.7

LEARNING_RATE = 0.001
GAMMA = 0.9
FINAL_EXPLORATION_RATE = 0.001

# Actor-Critic approach for maximizing entropy in order to stabilize the policy
ENTROPY_WEIGHT = 1.0

# Rewards (try to keep it rewards between [-1; 1])
SUCCESS_REWARD = 1
LEGAL_REWARD = -0.05
DISTANCE_REDUCTION_REWARD = 0.01
ILLEGAL_REWARD = -1


# Action/Observation Space
if SIMPLE_GAME:
    NUM_SUBBUILDINGS = 4
else:
    # NUM_SUBBUILDINGS may be reduced to 8 (instead of 16). mine/combiners are not crucial (at least for now)
    NUM_SUBBUILDINGS = 8
NUM_DIRECTIONS = 4
NUM_ACTIONS = NUM_DIRECTIONS * NUM_SUBBUILDINGS
