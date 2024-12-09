from best_ball_jepa_v3 import BallJEPA
from dataset import *
from best_jepa_config import *

wall_data = create_wall_dataloader(f"{DATA_PATH}/train")

# Phase 1 Training
ball_jepa = BallJEPA(
    ENCODER_LAYER_SIZES,
    BORDER_ENCODER_LAYER_SIZES,
    ENCODER_FINAL_ACTIVATION,
    ENCODER_LEAKY_RELU_MULT,
    PRED_LAYER_SIZES_P1,
    PRED_LAYER_SIZES_P2,
    PRED_FINAL_ACTIVATION,
    PRED_LEAKY_RELU_MULT,
    incl_border_encs=False,
    load=False
)
ball_jepa.train_model(wall_data, training_phase=1, num_epochs=5, lr=1e-2, lambda_=5e-5, pow=0.75)

"""
# uncomment below when train for phase 2, and comment out the phase 1 portion
# Phase 2 Training
ball_jepa = BallJEPA(
    ENCODER_LAYER_SIZES,
    BORDER_ENCODER_LAYER_SIZES,
    ENCODER_FINAL_ACTIVATION,
    ENCODER_LEAKY_RELU_MULT,
    PRED_LAYER_SIZES_P1,
    PRED_LAYER_SIZES_P2,
    PRED_FINAL_ACTIVATION,
    PRED_LEAKY_RELU_MULT,
    incl_border_encs=True,
    load=True
)
ball_jepa.train_model(wall_data, training_phase=2, num_epochs=20, lr=1e-3, lambda_=7.5e-5, pow=0.75)
"""
