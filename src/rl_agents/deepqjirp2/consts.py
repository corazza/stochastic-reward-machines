INPUT_SHAPE = (84, 84)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                   # Number of samples the agent learns from at once
LEARNING_RATE = 2.5e-4
USE_PER = True

DISCOUNT_FACTOR = 0.99            # Gamma, how much to discount future rewards
MIN_REPLAY_BUFFER_SIZE = 10*int(50000 / 6)    # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = int(1e6)           # The maximum size of the replay buffer

UPDATE_FREQ = 40                   # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 50000         # Number of actions between when the target network is updated

MAX_EPISODE_LENGTH = 18000        # replaced by done? # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes
FRAMES_BETWEEN_EVAL = 30000       # Number of frames between evaluations
EVAL_LENGTH = 10000               # Number of frames to evaluate for


PRIORITY_SCALE = 0.7               # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities

TOTAL_TIMESTEPS = int(1e7 / 6)       # Total number of frames to train for
