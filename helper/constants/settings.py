GYM_ID = "Profit"
PRINT_ROUND_LOG = False

# Replace Deposit/Factory with single celled variant + only 4 placable buildings: conveyors are 2x1
SIMPLE_GAME = True

MODEL_ID = "DQN"  # "DQN" or "A-C"
# Gradually increase Environment size in order to train convolutional layers one after the other
TRANSFER_LEARNING = False
# continue training of partially trained model (currently not implemented)
CONTINUE_TRAINING = False

# Model
NUM_CONV_FILTERS = 256
KERNEL_SIZE = 3
NUM_FEATURES = 128

# Freeze all transfered convolutional layers except last (allow for some adjustments of the larger grid size)
RETRAIN_LAST_CONV_LAYER = False

# Hyperparameters
MAX_EPISODES = 10000
PRE_TRAIN_EPISODES = MAX_EPISODES // 10
FINE_TUNE_EPISODES = MAX_EPISODES // 10
MAX_STEPS_EACH_EPISODE = 300
MAX_OBSTACLE_PROBABILITY = 0.4
# what is the minimum reward before increasing difficulty?
INCREASE_DIFFICULTY_AT = 0.0
# at what reward level should the difficulty be at its maximum?
MAX_DIFFICULTY_AT = 0.7

LEARNING_RATE = 0.001
FINAL_EXPLORATION_RATE = 0.001
GAMMA = 0.9
ENTROPY_WEIGHT = 0.05

# Rewards (try to keep it rewards between [-1; 1])
SUCCESS_REWARD = 1
LEGAL_REWARD = -0.1
DISTANCE_REDUCTION_REWARD = 0.1
ILLEGAL_REWARD = -1


# Action/Observation Space (only change when the game structure has changed)
FIELD_OF_VISION_SIZE = 7
if SIMPLE_GAME:
    NUM_SUBBUILDINGS = 4
else:
    # NUM_SUBBUILDINGS may be reduced to 8 (instead of 16). mine/combiners are not crucial (at least for now)
    NUM_SUBBUILDINGS = 16
NUM_DIRECTIONS = 4
NUM_ACTIONS = NUM_DIRECTIONS * NUM_SUBBUILDINGS
# NUM_CHANNELS should match channel-dimension of profit_gym.grid_to_observation()
NUM_CHANNELS = 1
