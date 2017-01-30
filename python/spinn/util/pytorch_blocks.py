from functools import partial
import argparse
import itertools

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


def HeKaimingInit(shape, real_shape=None):
    # Calculate fan-in / fan-out using real shape if given as override
    fan = real_shape or shape

    return np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                            size=shape)


class BaseSentencePairTrainer(object):

    def __init__(self, model, gpu=-1, **kwargs):
        self.model = model

        # TODO: Make these persistent
        self.model.best_dev_error = 0.0
        self.model.step = 0

    def init_optimizer(self, lr=0.01, l2_lambda=0.0, **kwargs):
        relevant_params = [w for w in self.model.parameters() if w.requires_grad]
        self.optimizer = optim.SGD(relevant_params, lr=lr, weight_decay=l2_lambda)

    def update(self):
        self.optimizer.step()

    def reset(self):
        self.optimizer.zero_grad()

    def forward(self, x_batch, y_batch=None, train=True,
                use_reinforce=False, rl_style="zero-one", rl_baseline="ema",
                use_internal_parser=False, validate_transitions=True, use_random=False):
        assert "sentences" in x_batch and "transitions" in x_batch, \
            "Input must contain dictionary of sentences and transitions."

        sentences =  torch.from_numpy(x_batch["sentences"]).long()
        transitions = x_batch["transitions"]

        if y_batch is not None:
            y_batch = torch.from_numpy(y_batch).long()

        if train:
            self.model.train()
        else:
            self.model.eval()

        ret = self.model(sentences, transitions, y_batch, train=train,
            use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline,
            use_internal_parser=use_internal_parser, validate_transitions=validate_transitions, use_random=use_random)
        y = ret[0]
        return ret

    def save(self, filename, step, best_dev_error):
        self.model.step = step
        self.model.best_dev_error = best_dev_error
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
        return self.model.step, self.model.best_dev_error