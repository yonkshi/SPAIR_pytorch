import os

BATCH_SIZE = 32
INPUT_IMAGE_SHAPE = [1, 128, 128] # originally 128 128 TODO change me back

DEFAULT_MLP_TOPOLOGY = [100, 100]
DEFAULT_BACKBONE_TOPOLOGY = [
    dict(filters=128, kernel_size=4, stride=3),
    dict(filters=128, kernel_size=4, stride=2),
    dict(filters=128, kernel_size=4, stride=2),
    dict(filters=128, kernel_size=1, stride=1),
    dict(filters=128, kernel_size=1, stride=1),
    dict(filters=128, kernel_size=1, stride=1),
]
CONV_OBJECT_ENCODER_TOPOLOGY = [ # Decoder is the opposite topology
    dict(filters=32, kernel_size=4, stride=2), # (32, 13, 13)
    dict(filters=32, kernel_size=3, stride=2), # (32, 5, 5)
    dict(filters=32, kernel_size=3, stride=2), # (32, 2, 2)
    dict(filters=32, kernel_size=1, stride=1), # (32, 2, 2)
]

N_BACKBONE_FEATURES = 100

N_PASSTHROUGH_FEATURES = 100

# Object attribute dimensions
N_ATTRIBUTES = 50
N_CONTEXT_DIM = 4 + N_ATTRIBUTES + 1 + 1

# Defines the range in which neighbouring cells are sampled to compute lateral context
N_LOOKBACK = 1

OBJECT_SHAPE = [28,28]
ANCHORBOX_SHAPE = [48, 48] # TODO 48 x 48


# Bounding box stuff, it's the maximum range relative to anchor box
MAX_YX = 1.5
MIN_YX = -0.5
MAX_HW = 1.0
MIN_HW = 0.0


# VAE Priors, used to KL computation {name:[mean, std]}
PRIORS = {
    'cy_logit':[0., 1.],
    'cx_logit':[0., 1.],
    'height_logit':[-0.9542425094, 0.5], # Larger prior for 28 x 28
    'width_logit':[-0.9542425094, 0.5],
    'attr':[0., 1.],
    'depth_logit':[0., 1.],
}

# Beta factor for Beta VAE
VAE_BETA = 1

# training wheels
LATENT_VAR_TRAINING_WHEEL_PARAM = dict(start = 1.0, # 1.0
                                       end = 0.0,
                                       decay_rate = 0.0,
                                       decay_step = 1000.,
                                       staircase = True)

# Dyanmic prior used by the object presence latent variable
OBJ_PRES_COUNT_LOG_PRIOR = dict(start = 1000000.0,
                                       end = 0.0125,
                                       decay_rate=0.1,
                                       decay_step = 1000.,
                                       log_space = True)


# Decoder bias:

OBJ_LOGIT_SCALE = 2.0
ALPHA_LOGIT_SCALE = 0.1
ALPHA_LOGIT_BIAS = 5.0


# environment variables

IS_LOCAL = 'LOCAL' in os.environ