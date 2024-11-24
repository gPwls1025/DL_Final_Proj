import torch
from torch import nn
from torch.optim import Adam
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from dataset import *
from models import MLP

from tqdm import tqdm
from jepa_config import WEIGHT_PATH, DISPLAY_LEN, IMAGE_SIZE, TOL, BORDER_MASK_THRESH, BORDER_SCALE

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def augment(states):
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    states = TF.rotate(states, angle)
    
    # Random flip
    if random.random() > 0.5:
        states = TF.hflip(states)
    if random.random() > 0.5:
        states = TF.vflip(states)
    
    # Small random translation
    max_dx, max_dy = 2, 2
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)
    states = TF.affine(states, angle=0, translate=(dx, dy), scale=1, shear=0)
    
    # Reduced Gaussian noise
    noise = torch.randn_like(states) * 0.005
    states = states + noise
    states = torch.clamp(states, 0, 1)
    
    return states

class BallJEPA(nn.Module):
    def __init__(
        self, encoder_layer_sizes, encoder_final_activation, encoder_leaky_relu_mult,
        pred_layer_sizes, pred_final_activation, pred_leaky_relu_mult,
        incl_border_encs=False, load=False
    ):
        super(BallJEPA, self).__init__()

        self.incl_border_encs = incl_border_encs
        self.repr_dim = 2

        self.encoder_layer_sizes = encoder_layer_sizes
        self.encoder_final_activation = encoder_final_activation
        self.encoder_leaky_relu_mult = encoder_leaky_relu_mult

        self.pred_layer_sizes = pred_layer_sizes
        self.pred_final_activation = pred_final_activation
        self.pred_leaky_relu_mult = pred_leaky_relu_mult

        if incl_border_encs:
            self.repr_dim += 2

        self.image_encoder = MLP(
            layer_sizes=encoder_layer_sizes,
            final_activation=encoder_final_activation,
            leaky_relu_mult=encoder_leaky_relu_mult,
        ).to("cuda")

        self.projector = MLP(
                layer_sizes=[self.repr_dim * 2, 64, 64, 64],
                final_activation=None,
                leaky_relu_mult=pred_leaky_relu_mult
            ).to("cuda")

        self.predictor = MLP(
            layer_sizes=pred_layer_sizes,
            final_activation=pred_final_activation,
            leaky_relu_mult=pred_leaky_relu_mult
        ).to("cuda")

        if load:
            self.load_state_dict(torch.load(WEIGHT_PATH, weights_only=True))

    def _preprocess_border_images(self, border_images):
        mask_x = torch.arange(IMAGE_SIZE).unsqueeze(0).repeat(IMAGE_SIZE, 1) / IMAGE_SIZE
        mask_y = torch.transpose(mask_x, 1, 0)
        mask = torch.where((abs(mask_x - 0.5) > BORDER_MASK_THRESH) | (abs(mask_y - 0.5) > BORDER_MASK_THRESH), 0, 1).unsqueeze(0).repeat(border_images.shape[0],1,1).to("cuda")
        border_images *= mask
        border_images /= BORDER_SCALE

        return border_images.reshape(border_images.shape[0], border_images.shape[1] * border_images.shape[2])

    def _produce_encodings(self, image_batch):
        ball_images, border_images = image_batch[:,0,:,:], image_batch[:,1,:,:]

        ball_images = ball_images.reshape(ball_images.shape[0], ball_images.shape[1] * ball_images.shape[2])
        border_images = self._preprocess_border_images(border_images)

        ball_encodings = self.image_encoder(ball_images)
        border_encodings = self.image_encoder(border_images)

        return ball_encodings, border_encodings

    def _predict_next_state(self, ball_encodings, border_encodings, actions, incl_mults=True, use_border_encs=True):
        actions /= IMAGE_SIZE
        inputs = torch.cat((
            ball_encodings,
            border_encodings if use_border_encs else torch.zeros(*border_encodings.shape).to("cuda"),
            actions
        ), dim=1)

        mults = self.predictor(inputs)
        return ball_encodings + (1 - mults if incl_mults else 1) * actions, mults

    def vicreg_loss(self, z1, z2, var_weight=25.0, inv_weight=25.0, cov_weight=1.0, eps=1e-4):
        # Invariance loss
        sim_loss = inv_weight * F.mse_loss(z1, z2)
        
        # Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = var_weight * (torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2)))
        
        # Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = cov_weight * (off_diagonal(cov_z1).pow_(2).sum() / z1.shape[1] + 
                                off_diagonal(cov_z2).pow_(2).sum() / z2.shape[1])
        
        return sim_loss + std_loss + cov_loss

    def forward(self, states, actions, incl_mults=True):
        init_states = states[:,0,:,:-1,:-1]
        ball_encs, border_encs = self._produce_encodings(init_states)

        out_encs = [torch.cat((ball_encs,border_encs) if self.incl_border_encs else (ball_encs,), dim=1)]

        for idx in range(1, states.shape[1]):
            ball_encs, _ = self._predict_next_state(ball_encs, border_encs, actions[:,idx-1,:], incl_mults=incl_mults)
            out_encs.append(torch.cat((ball_encs,border_encs) if self.incl_border_encs else (ball_encs,), dim=1))

        return torch.stack(out_encs)

    def train_model(self, train_data, training_phase, num_epochs=5, lr=1e-3, lambda_=0, pow=1):
        if training_phase == 1:
            params = list(self.image_encoder.parameters()) + list(self.predictor.parameters())
            self.image_encoder.train()
        elif training_phase == 2:
            # Initialize predictor weights
            for name, param in self.predictor.named_parameters():
                if 'weight' in name or 'bias' in name:
                    nn.init.uniform_(param,-0.25,0.25)
            params = list(self.predictor.parameters()) + list(self.projector.parameters())
            self.image_encoder.eval()
            self.predictor.train()

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
                actions = batch.actions.reshape(*actions_shape).to("cuda")

                states_in = states_in[:,:,:-1,:-1]
                states_out = states_out[:,:,:-1,:-1]

                ball_enc, border_enc = self._produce_encodings(states_in)
                preds, mults = self._predict_next_state(ball_enc, border_enc, actions, use_border_encs=(training_phase > 1))
                targets, _ = self._produce_encodings(states_out)

                loss = criterion(preds, targets)
                losses.append(loss.item())
                loss.backward(retain_graph=True)

                secondary_loss = lambda_ * torch.pow(torch.clip(mults, min=TOL, max=1), pow).mean()
                secondary_losses.append(secondary_loss.item())
                secondary_loss.backward(retain_graph=True)

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

                optimizer.step()

                targets = targets.view(*targets_shape)
                stds.append(torch.tensor([t.std(dim=0).mean().item() for t in targets]).mean() + TOL)

                desc = f"Avg RMSE = {round(torch.sqrt(torch.mean(torch.tensor(losses[-DISPLAY_LEN:]))).item(), 6)}, "
                desc += f"Targets Stdev = {round(torch.mean(torch.tensor(stds[-DISPLAY_LEN:])).item(), 6)}, "
                desc += f"R2 = {round(1 - (torch.sqrt(torch.mean(torch.tensor(losses[-DISPLAY_LEN:]))).item() / torch.mean(torch.tensor(stds[-DISPLAY_LEN:])).item()) ** 2, 4)}, "
                desc += f"Avg Secondary Loss = {round(torch.mean(torch.tensor(secondary_losses[-DISPLAY_LEN:])).item() / lambda_, 4)}"
                if training_phase == 2:
                    desc += f", Avg VICReg Loss = {round(torch.mean(torch.tensor(vicreg_losses[-DISPLAY_LEN:])).item(), 4)}"
                pbar.set_description(desc)

            torch.save(self.state_dict(), WEIGHT_PATH)

        self.image_encoder.eval()
        self.predictor.eval()