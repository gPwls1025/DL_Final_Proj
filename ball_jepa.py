import torch
from torch import nn
from torch.optim import Adam
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dataset import *
from models import CNN, MLP, IMAGE_SIZE, TOL

from tqdm import tqdm
import random

BALL_CNN_PATH = "ball_cnn.pth"
BORDER_CNN_PATH = "border_cnn.pth"
PREDICTOR_PATH = "predictor.pth"

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def augment(states):
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    states = TF.rotate(states, angle)

    # Random flip
    if random.random() > 0.3:
        states = TF.hflip(states)
    if random.random() > 0.3:
        states = TF.vflip(states)

    # Small random translation
    max_dx, max_dy = 3, 3
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    states = TF.affine(states, angle=0, translate=(dx, dy), scale=1, shear=0)

    # Add small Gaussian noise
    noise = torch.randn_like(states) * 0.005
    states = states + noise
    states = torch.clamp(states, 0, 1)

    return states

class BallJEPA(nn.Module):
    def __init__(
        self, ball_cnn_channel_sizes, ball_cnn_out_dim, ball_cnn_pool_size, ball_cnn_kernel_size, ball_cnn_final_activation,
        border_cnn_channel_sizes, border_cnn_out_dim, border_cnn_pool_size, border_cnn_kernel_size, border_cnn_final_activation,
        pred_layer_sizes, pred_final_activation
    ):
        super(BallJEPA, self).__init__()
        self.repr_dim = ball_cnn_out_dim

        self.ball_cnn = CNN(
            channel_sizes=ball_cnn_channel_sizes,
            out_dim=ball_cnn_out_dim,
            pool_size=ball_cnn_pool_size,
            kernel_size=ball_cnn_kernel_size,
            final_activation=ball_cnn_final_activation
        ).to("cuda")
        self.border_cnn = CNN(
            channel_sizes=border_cnn_channel_sizes,
            out_dim=border_cnn_out_dim,
            pool_size=border_cnn_pool_size,
            kernel_size=border_cnn_kernel_size,
            final_activation=border_cnn_final_activation
        ).to("cuda")

        # Add a new MLP after encoders to project the representations into a space where VICReg loss will be applied
        self.projector = MLP(
            layer_sizes=[ball_cnn_out_dim + border_cnn_out_dim, 2048, 2048, 2048],
            final_activation=nn.ReLU()
        ).to("cuda")

        self.predictor = MLP(
            layer_sizes=pred_layer_sizes,
            final_activation=pred_final_activation
        ).to("cuda")

        # 2nd predictor for middle wall prediction
        """
        ball_cnn_out_dim: output from ball CNN (ball encoding)
        border_cnn_out_dim: oujtput from boarder CNN (border encoding)
        + 2: including action input (2d; x, y movement)
        32: hidden layer size
        2: output dimension
        """
        self.wall_predictor = MLP(
            layer_sizes=[ball_cnn_out_dim + border_cnn_out_dim + 2, 32, 2],
            final_activation=torch.tanh
        ).to("cuda")

        self.training_phase = 1

    def load_weights(self):
        self.ball_cnn.load_state_dict(torch.load(BALL_CNN_PATH))
        self.border_cnn.load_state_dict(torch.load(BORDER_CNN_PATH))
        self.predictor.load_state_dict(torch.load(PREDICTOR_PATH))

    def save_weights(self):
        torch.save(self.ball_cnn.state_dict(), BALL_CNN_PATH)
        torch.save(self.border_cnn.state_dict(), BORDER_CNN_PATH)
        torch.save(self.predictor.state_dict(), PREDICTOR_PATH)

    def _produce_encodings(self, image_batch):
        ball_images, border_images = image_batch[:,0,:,:].unsqueeze(1), image_batch[:,1,:,:].unsqueeze(1)

        ball_encodings = self.ball_cnn(ball_images)
        border_encodings = self.border_cnn(border_images)

        return ball_encodings, border_encodings

    def _predict_next_state(self, ball_encodings, border_encodings, actions):
        # first predictor
        inputs = torch.cat((ball_encodings, actions, border_encodings), dim=1)
        mults = self.predictor(inputs)

        initial_prediction = ball_encodings + mults * actions

        if self.training_phase == 2:
            # second predictor 
            wall_inputs = torch.cat((initial_prediction, border_encodings, actions), dim=1)
            wall_adjustment = self.wall_predictor(wall_inputs)
            final_prediction = initial_prediction + wall_adjustment * actions
        else:
            # phase 1
            final_prediction = initial_prediction

        return final_prediction, mults

    def vicreg_loss(self, z1, z2, var_weight=25.0, inv_weight=25.0, cov_weight=1, eps=1e-4):
        # Invariance loss
        sim_loss = inv_weight * F.mse_loss(z1, z2)
        print("Invariance loss:", sim_loss)
        
        # Variance loss (increased weight to penalize low variance)
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)  # Added epsilon for stability
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = var_weight * (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2)))
        print("Variance loss:", std_loss)

        # Covariance loss (with epsilon and weight)
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)  
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = cov_weight * (off_diagonal(cov_z1).pow_(2).sum() / z1.shape[1] + 
                                off_diagonal(cov_z2).pow_(2).sum() / z2.shape[1])
        print("Covariance loss:", cov_loss)

        return sim_loss + std_loss + cov_loss


    def forward(self, states, actions, incl_border_encodings=False):
        init_states = states[:,0,:,:,:]
        ball_encs, border_encs = self._produce_encodings(init_states)

        out_encs = [torch.cat((ball_encs,border_encs) if incl_border_encodings else (ball_encs,), dim=1)]

        for idx in range(1, states.shape[1]):
            ball_encs, _ = self._predict_next_state(ball_encs, border_encs, actions[:,idx-1,:] / IMAGE_SIZE)
            out_encs.append(torch.cat((ball_encs,border_encs) if incl_border_encodings else (ball_encs,), dim=1))

        return torch.stack(out_encs)

    def train_model(self, train_data, training_phase, num_epochs=5, lr=1e-3, lambda_=0):
        # Parameter selection based on training phase
        if training_phase == 1:
            params = self.ball_cnn.parameters()
        elif training_phase == 2:
            self.load_weights()
            params = list(self.ball_cnn.parameters()) + list(self.border_cnn.parameters()) + list(self.predictor.parameters()) + list(self.wall_predictor.parameters()) + list(self.projector.parameters())

        self.ball_cnn.train()
        self.border_cnn.train()
        self.predictor.train()
        self.wall_predictor.train()

        optimizer = Adam(params, lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}")
            pbar = tqdm(train_data)
            losses = []
            secondary_losses = []
            vicreg_losses = []
            stds = []

            for batch in pbar:
                optimizer.zero_grad()

                states_shape = (
                    batch.states.shape[0] * (batch.states.shape[1] - 1),
                    batch.states.shape[2],
                    batch.states.shape[3],
                    batch.states.shape[4]
                )
                actions_shape = (batch.actions.shape[0] * batch.actions.shape[1], batch.actions.shape[2])

                states_in = batch.states[:,:-1,:,:,:].reshape(*states_shape).to("cuda")
                states_out = batch.states[:,1:,:,:,:].reshape(*states_shape).to("cuda")
                actions = batch.actions / IMAGE_SIZE
                actions = actions.reshape(*actions_shape).to("cuda")

                states_in = states_in[:,:,:-1,:-1]
                states_out = states_out[:,:,:-1,:-1]

                # Basic operations for both phases
                ball_enc, border_enc = self._produce_encodings(states_in)
                preds, mults = self._predict_next_state(ball_enc, border_enc, actions)
                targets, _ = self._produce_encodings(states_out)
                # measures how well the model predicts the next state -> ball prediction
                loss = criterion(preds, targets)
                # Penalizes when multiplication factors (mults) deviate from 1
                secondary_loss = lambda_ * torch.pow(1 - mults, 2).mean()

                losses.append(loss.item())
                secondary_losses.append(secondary_loss.item())
                stds.append(targets.std(dim=0).mean().item() + TOL)

                # Backward passes for basic operations
                loss.backward(retain_graph=True)
                secondary_loss.backward(retain_graph=True)

                # Add VICReg only in phase 2
                if training_phase == 2:
                    states_in_aug1 = augment(states_in)
                    states_in_aug2 = augment(states_in)
                    ball_enc1, border_enc1 = self._produce_encodings(states_in_aug1)
                    ball_enc2, border_enc2 = self._produce_encodings(states_in_aug2)
                    z1 = self.projector(torch.cat((ball_enc1, border_enc1), dim=1))
                    z2 = self.projector(torch.cat((ball_enc2, border_enc2), dim=1))

                    vicreg_loss = self.vicreg_loss(z1, z2)
                    vicreg_loss.backward()
                    vicreg_losses.append(vicreg_loss.item())
                else:
                    vicreg_losses.append(0.0)
                
                optimizer.step()

                desc = f"Avg RMSE = {round(torch.sqrt(torch.mean(torch.tensor(losses[-100:]))).item(), 6)}, "
                desc += f"Targets Stdev = {round(torch.mean(torch.tensor(stds[-100:])).item(), 6)}, "
                desc += f"R2 = {round(1 - (torch.sqrt(torch.mean(torch.tensor(losses[-100:]))).item() / torch.mean(torch.tensor(stds[-100:])).item()) ** 2, 4)}, "
                desc += f"Avg Secondary Loss = {round(torch.mean(torch.tensor(secondary_losses[-100:])).item() / lambda_, 4)}"
                if training_phase == 2:
                    desc += f", Avg VICReg Loss = {round(torch.mean(torch.tensor(vicreg_losses[-100:])).item(), 4)}"
                pbar.set_description(desc)

            self.save_weights()

        self.ball_cnn.eval()
        self.border_cnn.eval()
        self.predictor.eval()
        self.wall_predictor.eval()