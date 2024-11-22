from ball_jepa import BallJEPA
from dataset import *
from jepa_config import *

wall_data = create_wall_dataloader(f"{DATA_PATH}/train")
ball_jepa = BallJEPA(
    BALL_CNN_CHANNEL_SIZES,
    BALL_CNN_OUT_DIM,
    BALL_CNN_POOL_SIZE,
    BALL_CNN_KERNEL_SIZE,
    BALL_CNN_FINAL_ACTIVATION,
    BORDER_CNN_CHANNEL_SIZES,
    BORDER_CNN_OUT_DIM,
    BORDER_CNN_POOL_SIZE,
    BORDER_CNN_KERNEL_SIZE,
    BORDER_CNN_FINAL_ACTIVATION,
    PRED_LAYER_SIZES,
    PRED_FINAL_ACTIVATION
)
ball_jepa.train_model(wall_data, num_epochs=20, lr=1e-3, lambda_=5e-5)
ball_jepa.train_model(wall_data, num_epochs=10, lr=5e-4, lambda_=5e-5)  # Phase 2
