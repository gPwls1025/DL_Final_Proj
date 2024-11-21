import numpy as np

import matplotlib.pyplot as plt
from dataset import *

import inspect
from jepa_config import *
from ball_jepa import BallJEPA
DATA_PATH = "/scratch/DL24FA"

wall_data = create_wall_dataloader(
    data_path=f"{DATA_PATH}/probe_normal/train",
    probing=True,
    device='cuda',
    train=True,
)
model = BallJEPA(
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
model.load_weights()
wall_data = iter(wall_data)

for k in range(10):
    sample = next(wall_data)
    print(sample.actions[2])
    print(sample.locations[2])
    print(model(sample.states, sample.actions)[:,0,:2])
    print(model(sample.states, sample.actions).transpose(1,0).shape)
    print(sample.locations.shape)
    plt.scatter(model(sample.states, sample.actions).transpose(1,0)[:,:,1].cpu().detach().numpy(),sample.locations[:,:,1].cpu().detach().numpy())
plt.savefig('res.jpg')
"""
print(sample.states.shape)
print(sample.locations.shape)
print(sample.actions.shape)

for action in sample.actions[0]:
    print(action)

fig, ax = plt.subplots(5,4, figsize=(15,12))

for j in range(5):
    for k in range(4):
        ax[j][k].imshow(sample.states[0,4*j+k,0,:].cpu().numpy())

        if 4*j+k == 16:
            break

plt.savefig('test.jpg')

for j in range(5):
    for k in range(4):
        ax[j][k].imshow(sample.states[0,4*j+k,1,:].cpu().numpy())

        if 4*j+k == 16:
            break

plt.savefig('test2.jpg')

for j in range(5):
    for k in range(4):
        ax[j][k].imshow(
            sample.states[0,4*j+k,1,:].cpu().numpy()+
            sample.states[0,4*j+k,0,:].cpu().numpy()
        )

        if 4*j+k == 16:
            break
"""
plt.savefig('test3.jpg')