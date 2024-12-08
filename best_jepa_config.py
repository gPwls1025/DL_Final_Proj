import torch

IMAGE_SIZE = 64
TOL = 1e-6
DISPLAY_LEN = 100

BORDER_MASK_THRESH = 0.425
BORDER_SCALE = 16

DATA_PATH = "/scratch/DL24FA"
#WEIGHT_PATH = "model_weights.pth"
PHASE1_WEIGHT_PATH = "model_weights_phase1.pth"
PHASE2_WEIGHT_PATH = "model_weights_phase2.pth"

ENCODER_LAYER_SIZES = [
    int(IMAGE_SIZE ** 2),
    2
]

# Border encoder outputs higher dim
BORDER_ENCODER_LAYER_SIZES = [
    int(IMAGE_SIZE ** 2),
    64  # Higher dimensional output
]

ENCODER_FINAL_ACTIVATION = torch.sigmoid
ENCODER_LEAKY_RELU_MULT = 1e-3

"""
PRED_LAYER_SIZES = [
    6,
    64,64,64,64, 
    1
]
"""
# For Phase 1
PRED_LAYER_SIZES_P1 = [
    4,  # 2 (ball) + 2 (actions)
    64, 64, 64, 64,
    1
]

# For Phase 2
PRED_LAYER_SIZES_P2 = [
    68,  # 2 (ball) + 64 (border) + 2 (actions)
    64, 64, 64, 64,
    1
]

PRED_FINAL_ACTIVATION = torch.sigmoid
PRED_LEAKY_RELU_MULT = 1e-3
