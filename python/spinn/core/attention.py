import itertools

import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import LSTMState, Embed, MLP, Linear, LSTM
from spinn.util.blocks import reverse_tensor
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm

from spinn.util.catalan import ShiftProbabilities
from spinn.data import T_SHIFT, T_REDUCE, T_SKIP


class SequenceAttention(nn.Module):
    """Attend over an input sequence using a query.

    Attention uses the following function:

    .. math::

        \begin{array}{ll}
        \beta_i = U \tanh(W_x x_i + W_q q)
        \alpha = softmax(\beta)
        h = \sum_i \alpha_i x_i
        \end{array}

    where :math:`x_i` are the features of the input sequence at index `i` and :math:`q` is the query.

    Args:
        input_size: The number of features in the hidden states of the input `x`.
        query_size: The number of features in the query `q`.
        hidden_size: The number of features in the hidden dimension of the attention layer.

    Inputs: input, query
        - **input** (batch, seq_len, input_size): tensor containing the features of the input sequence `x`.
        - **query** (batch, query_size): tensor containing the query `q` for the input sequence.

    Outputs: output, alpha
        - **output** (batch, input_size): tensor containing
          the weighted sum the input sequence's features.
        - **alpha** (batch, seq_len): tensor containing the scores used to compute the weighted sum.

    Attributes:
        weight_W_x : the learnable input-hidden weights of shape `(input_size x hidden_size)`
        weight_W_q : the learnable input-hidden weights of shape `(query_size x hidden_size)`
        weight_U   : the learnable input-hidden weights of shape `(hidden_size x 1)`
        bias_W_x   : the learnable input-hidden bias of shape `(hidden_size)`
        bias_W_q   : the learnable input-hidden bias of shape `(hidden_size)`
        bias_U     : the learnable input-hidden bias of shape `(1)`

    Examples::

        >>> attn = Attention(10, 3, 5)
        >>> input = Variable(torch.randn(2, 5, 10))
        >>> query = Variable(torch.randn(2, 3))
        >>> output, alpha = attn(input, query)
    """

    def __init__(self, input_size, query_size, hidden_size):
        super(SequenceAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.hidden_size = hidden_size

        self.W_x = nn.Linear(input_size, hidden_size)
        self.W_q = nn.Linear(query_size, hidden_size)
        self.U = nn.Linear(hidden_size, 1)

    def forward(self, input_sequence, query):
        hidden_size = self.hidden_size
        batch_size, seq_len, input_size = input_sequence.size()

        hq = self.W_q(query) # pre-compute query
        hq_broadcast = hq.unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        hq_flatten = hq_broadcast.contiguous().view(batch_size * seq_len, hidden_size)

        x = input_sequence.view(batch_size * seq_len, input_size)
        hx = self.W_x(x)

        beta = self.U(F.tanh(hx + hq_flatten))
        alpha = F.softmax(beta.view(batch_size, seq_len))
        alpha_broadcast = alpha.unsqueeze(2).expand(batch_size, seq_len, input_size)

        h = input_sequence * input_sequence
        h = h.sum(1).squeeze()

        return h, alpha
