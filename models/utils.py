import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import math
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def seed_everything(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if torch.cuda.is_available():
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg):    
    src_mask = (src != 0).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr