import os
import logging
import numpy as np
import random

import torch

def set_seed(seed):
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - [%(filename)s:%(lineno)d]: %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model, optimizer, epoch, loss,checkpoint_path):
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def tensor_to_list(tensor):
    if tensor.dim() > 1:
        tensor = tensor.view(-1)
    return tensor.cpu().detach().numpy().tolist()
