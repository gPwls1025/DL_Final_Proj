from ball_jepa import BallJEPA
from dataset import *
from jepa_config import *

wall_data = create_wall_dataloader(f"{DATA_PATH}/train")
ball_jepa = BallJEPA(
    ENCODER_LAYER_SIZES,
    ENCODER_FINAL_ACTIVATION,
    ENCODER_LEAKY_RELU_MULT,
    PRED_LAYER_SIZES,
    PRED_FINAL_ACTIVATION,
    PRED_LEAKY_RELU_MULT,
    load=True
)
#ball_jepa.train_model(wall_data, training_phase=1, num_epochs=5, lr=1e-2, lambda_=5e-5, pow=0.75)
ball_jepa.train_model(wall_data, training_phase=2, num_epochs=20, lr=1e-3, lambda_=7.5-5, pow=0.75)