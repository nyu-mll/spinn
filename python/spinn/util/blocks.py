import numpy as np
import random
import math
import six

from functools import partial
import argparse
import itertools

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim



def HeKaimingInit(var):
    fan = var.size()
    weight = torch.from_numpy(np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
        size=fan).astype(np.float32))
    var.data.set_(weight)


def UniformInitializer(var, range):
    var.data.uniform_(-range, range)


def ZeroInitializer(var):
    var.data.fill_(0)


def StDevClosure(var):
    return [1. / math.sqrt(var.weight.size(1))]


def EmptyClosure(var):
    return []


def Linear(in_features, out_features, bias=True, closure=EmptyClosure, initializer=UniformInitializer, bias_initializer=ZeroInitializer):
    class _Linear(nn.Linear):
        def reset_parameters(self):
            config = closure(self)
            initializer(self.weight, *config)
            if self.bias is not None:
                bias_initializer(self.bias, *config)
    return _Linear(in_features, out_features, bias)


def to_cuda(var, gpu):
    if gpu >= 0:
        return var.cuda()
    return var


def select_item(var, index, gpu=-1):
    index_mask = index.view(-1, 1).repeat(1, var.size(1))
    mask = to_cuda(torch.range(0, var.size(1) - 1).long(), gpu)
    mask = mask.repeat(var.size(0), 1)
    mask = mask.eq(index_mask)
    return torch.masked_select(var, Variable(mask, volatile=var.volatile))


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
                use_internal_parser=False, validate_transitions=True):
        assert "sentences" in x_batch and "transitions" in x_batch, \
            "Input must contain dictionary of sentences and transitions."

        sentences =  x_batch["sentences"]
        transitions = x_batch["transitions"]

        if train:
            self.model.train()
        else:
            self.model.eval()

        ret = self.model(sentences, transitions, y_batch, train=train,
            use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline,
            use_internal_parser=use_internal_parser, validate_transitions=validate_transitions)
        y = ret[0]
        return ret

    def save(self, filename, step, best_dev_error):
        self.model.step = step
        self.model.best_dev_error = best_dev_error
        torch.save(self.model, filename)

    def load(self, filename):
        self.model = torch.load(filename)
        return self.model.step, self.model.best_dev_error


class HardGradientClipping(object):

    """Optimizer hook function for hard gradient clipping.

    This hook function limits gradient values within the boundary.

    Args:
        x_min (float): minimum gradient value.
        x_max (float): maximum gradient value.

    Attributes:
        x_min (float): minimum gradient value.
        x_max (float): maximum gradient value.

    """
    name = 'HardGradientClipping'

    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, opt):
        for param in opt.target.params():
            param.grad = F.clip(param.grad, self.x_min, self.x_max).data


def dropout(inp, ratio, train):
    if ratio > 0:
        return F.dropout(inp, ratio, train)
    return inp


def expand_dims(var, dim=0):
    sizes = list(var.size())
    sizes.insert(dim, 1)
    return var.view(*sizes)


def expand_along(var, mask):
    indexes = np.tile(np.arange(var.shape[0]).reshape(-1, 1), (1, mask.shape[1]))
    _mask = indexes[mask]
    return var[_mask]


def var_mean(x, axis=0):
    return F.sum(x) / x.shape[axis]


def is_train(var):
    return var.volatile == False


class LSTMState:
    """Class for intelligent LSTM state object.

    It can be initialized from either a tuple ``(c, h)`` or a single variable
    `both`, and provides lazy attribute access to ``c``, ``h``, and ``both``.
    Since the SPINN conducts all LSTM operations on GPU and all tensor
    shuffling on CPU, ``c`` and ``h`` are automatically moved to GPU while
    ``both`` is automatically moved to CPU.

    Args:
        inpt: Either a tuple of ~chainer.Variable objects``(c, h)`` or a single
        concatenated ~chainer.Variable containing both.

    Attributes:
        c (~chainer.Variable): LSTM memory state, moved to GPU if necessary.
        h (~chainer.Variable): LSTM hidden state, moved to GPU if necessary.
        both (~chainer.Variable): Concatenated LSTM state, moved to CPU if
            necessary.

    """
    def __init__(self, inpt):
        if isinstance(inpt, tuple):
            self._c, self._h = inpt
        else:
            self._both = inpt
            self.size = inpt.size(1) // 2

    @property
    def h(self):
        if not hasattr(self, '_h'):
            self._h = get_h(self._both, self.size)
        return self._h

    @property
    def c(self):
        if not hasattr(self, '_c'):
            self._c = get_c(self._both, self.size)
        return self._c

    @property
    def both(self):
        if not hasattr(self, '_both'):
            self._both = torch.cat(
                (self._c, self._h), 1)
        return self._both


def get_c(state, hidden_dim):
    return state[:, hidden_dim:]

def get_h(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return torch.cat([h, c], 1)


def bundle(lstm_iter):
    """Bundle an iterable of concatenated LSTM states into a batched LSTMState.

    Used between CPU and GPU computation. Reversed by :func:`~unbundle`.

    Args:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.

    Returns:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``.
    """
    if lstm_iter is None:
        return None
    lstm_iter = tuple(lstm_iter)
    if lstm_iter[0] is None:
        return None
    return LSTMState(torch.cat(lstm_iter, 0))


def unbundle(state):
    """Unbundle a batched LSTM state into a tuple of concatenated LSTM states.

    Used between GPU and CPU computation. Reversed by :func:`~bundle`.

    Args:
        state: :class:`~LSTMState` object, with ``c`` and ``h`` attributes
            each with shape ``(B,S)``, or an ``inpt`` to
            :func:`~LSTMState.__init__` that would produce such an object.

    Returns:
        lstm_iter: Iterable of ``B`` ~chainer.Variable objects, each with
            shape ``(1,2*S)``, consisting of ``c`` and ``h`` concatenated on
            axis 1.
    """
    if state is None:
        return itertools.repeat(None)
    if not isinstance(state, LSTMState):
        state = LSTMState(state)
    return torch.chunk(
        state.both, state.both.size(0), 0)


def treelstm(c_left, c_right, gates, use_dropout=False):
    hidden_dim = c_left.size(1)

    assert gates.size(1) == hidden_dim * 5, "Need to have 5 gates."

    def slice_gate(gate_data, i):
        return gate_data[:, i * hidden_dim:(i + 1) * hidden_dim]

    # Compute and slice gate values
    i_gate, fl_gate, fr_gate, o_gate, cell_inp = \
        [slice_gate(gates, i) for i in range(5)]

    # Apply nonlinearities
    i_gate = F.sigmoid(i_gate)
    fl_gate = F.sigmoid(fl_gate)
    fr_gate = F.sigmoid(fr_gate)
    o_gate = F.sigmoid(o_gate)
    cell_inp = F.tanh(cell_inp)

    # Compute new cell and hidden value
    i_val = i_gate * cell_inp
    dropout_rate = 0.1
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)


class Reduce(nn.Module):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to four inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model; the fourth is an optional
    attentional input.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
    """

    def __init__(self, size, tracker_size=None, use_tracking_in_composition=True):
        super(Reduce, self).__init__()
        self.left = nn.Linear(size, 5 * size)
        self.right = nn.Linear(size, 5 * size, bias=False)
        if tracker_size is not None and use_tracking_in_composition:
            self.track = nn.Linear(tracker_size, 5 * size, bias=False)

    def __call__(self, left_in, right_in, tracking=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two or three inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model. All are provided
        as iterables and batched internally into tensors.

        Args:
            left_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the left child of each node
                in the batch.
            right_in: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the right child of each node
                in the batch.
            tracking: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the tracker LSTM state of
                each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~chainer.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking = bundle(tracking)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in))
        return out
