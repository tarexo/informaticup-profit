GYM_ID = "Profit"
PRINT_ROUND_LOG = False

# Replace Deposit/Factory with single celled variant + only 4 placable buildings: conveyors are 2x1
SIMPLE_GAME = True

MODEL_ID = "A-C"  # "DQN" or "A-C"
# Gradually increase Environment size in order to train convolutional layers one after the other
TRANSFER_LEARNING = True
# continue training of partially trained model (currently not implemented)
CONTINUE_TRAINING = False

# Model
INITIAL_CONV_FILTERS = 128
FILTER_DECREASING = False
NUM_FEATURES = 64

# Freeze all transfered convolutional layers except last (allow for some adjustments of the larger grid size)
RETRAIN_LAST_CONV_LAYER = True

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.9

MAX_EPISODES = 100000
MAX_STEPS_EACH_EPISODE = 100
MAX_OBSTACLE_PROBABILITY = 0.5

# Rewards (try to keep it rewards between [-1; 1])
SUCCESS_REWARD = 1
LEGAL_REWARD = 0
ILLEGAL_REWARD = 0


# Action/Observation Space (only change when the game structure has changed)
if SIMPLE_GAME:
    NUM_SUBBUILDINGS = 4
else:
    # NUM_SUBBUILDINGS may be reduced to 8 (instead of 16). mine/combiners are not crucial (at least for now)
    NUM_SUBBUILDINGS = 16
NUM_DIRECTIONS = 4
NUM_ACTIONS = NUM_DIRECTIONS * NUM_SUBBUILDINGS
# NUM_CHANNELS should match channel-dimension of profit_gym.grid_to_observation()
NUM_CHANNELS = 3
