# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Module for managing model checkpoint saving and recovery, #
# as well as conversion between numpy and tensor based on   #
# the current device without setting it manually            #
#                                                           #
# (This module is meant for saving time during development  #
#  and debugging. Please make sure your models and tensors  #
#  are placed in the correct devices if used on your own    #
#  applications)                                            #
#                                                           #
# Author: Miguel Marcos                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import numpy as np

def save_checkpoint(save_path, model, optimizer, epoch=0, loss=0.0):
    """
    Save an instance of a model and its optimizer to the selected path
    """

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

def load_checkpoint(load_path, model, optimizer = None, gpu_device='cuda:0'):
    """
    Given a model, an (optional) optimizer and the path
    to load the checkpoint from, return them initialized.
    """

    if torch.cuda.is_available():
        device = torch.device(gpu_device)
    else:
        device = torch.device('cpu')

    cp = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(cp['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(cp['optimizer_state_dict'])
    loss = cp['loss']
    epoch = cp['epoch']

    return device, epoch, loss

def tensor_to_np(x):

    return x.cpu().detach().numpy()

def np_to_tensor(x, t=torch.float32, gpu_device='cuda:0'):

    if torch.cuda.is_available():
        t = torch.tensor(x).to(device=gpu_device, dtype=t)
    else:
        t = torch.tensor(x).to(device='cpu', dtype=t)

    return t