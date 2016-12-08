import numpy as np
import random

# Chainer imports
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions
from chainer import testing

from chainer.utils import type_check


def gradient_check(model, get_loss, rtol=0, atol=1e-2, to_check=10):
    epsilon = 1e-3
    cached_grads = [w.grad.ravel().copy() for (n,w) in model.namedparams()]
    checked = []

    # TODO: Only consider non-zero gradients.
    for (_, w), cached in zip(model.namedparams(), cached_grads):
        chosen = range(len(cached))
        random.shuffle(chosen)
        chosen = chosen[:to_check]
        for c in chosen:
            # Find Y1
            w.data.ravel()[c] += epsilon
            check_loss_1 = get_loss()

            # Find Y2
            w.data.ravel()[c] -= 2 * epsilon
            check_loss_2 = get_loss()

            # Reset Param
            w.data.ravel()[c] += epsilon

            # Check that: Gradient ~ (Y1 - Y2) / (2 * epsilon)
            estimate = (check_loss_1.data - check_loss_2.data) / (2 * epsilon)
            checked.append((estimate, cached[c]))

    estimates, grads = zip(*checked)
    estimates, grads = np.array(estimates), np.array(grads)
    testing.assert_allclose(estimates, grads, rtol=rtol, atol=atol, verbose=True), "Gradient check failed."


def l2_cost(model, l2_lambda):
    cost = 0.0
    for _, w in model.namedparams():
        cost += l2_lambda * F.sum(F.square(w))
    return cost


def flatten(l):
    if hasattr(l, '__len__'):
        return reduce(lambda x, y: x + flatten(y), l, [])
    else:
        return [l]


def the_gpu():
    return the_gpu.gpu

the_gpu.gpu = -1

# Chainer already has a method for moving a variable to/from GPU in-place,
# but that messes with backpropagation. So I do it with a copy. Note that
# cuda.get_device() actually returns the dummy device, not the current one
# -- but F.copy will move it to the active GPU anyway (!)
def to_gpu(var):
    return F.copy(var, the_gpu())


def to_cpu(var):
    return F.copy(var, -1)


def arr_to_gpu(arr):
    if the_gpu() >= 0:
        return cuda.to_gpu(arr)
    else:
        return arr


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
            self.size = inpt.data.shape[1] // 2

    @property
    def h(self):
        if not hasattr(self, '_h'):
            self._h = to_gpu(self._both[:, self.size:])
        return self._h

    @property
    def c(self):
        if not hasattr(self, '_c'):
            self._c = to_gpu(self._both[:, :self.size])
        return self._c

    @property
    def both(self):
        if not hasattr(self, '_both'):
            self._both = F.concat(
                (to_cpu(self._c), to_cpu(self._h)), axis=1)
        return self._both


def get_c(state, hidden_dim):
    return state[:, hidden_dim:]

def get_h(state, hidden_dim):
    return state[:, :hidden_dim]

def get_state(c, h):
    return F.concat([h, c], axis=1)


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
    try:
        return LSTMState(F.concat(lstm_iter, axis=0))
    except:
        import ipdb; ipdb.set_trace()


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
    return F.split_axis(
        state.both, state.both.data.shape[0], axis=0, force_tuple=True)


def treelstm(c_left, c_right, gates, use_dropout=False):
    hidden_dim = c_left.shape[1]

    assert gates.shape[1] == hidden_dim * 5, "Need to have 5 gates."

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
    if use_dropout:
        i_val = F.dropout(i_val, dropout_rate, train=i_val.volatile == False)
    c_t = fl_gate * c_left + fr_gate * c_right + i_val
    h_t = o_gate * F.tanh(c_t)

    return (c_t, h_t)



class CrossEntropyClassifier(Chain):
    def __init__(self, gpu=-1):
        super(CrossEntropyClassifier, self).__init__()
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        pred_type, y_type = in_types

        type_check.expect(
            pred_type.dtype == 'f',
            pred_type.ndim >= 1,
        )

        type_check.expect(
            y_type.dtype == 'i',
            y_type.ndim >= 1,
        )

    def __call__(self, y, y_batch, train=True):
        # BEGIN: Type Check
        in_data = tuple([x.data for x in [y, y_batch]])
        in_types = type_check.get_types(in_data, 'in_types', False)
        self.check_type_forward(in_types)
        # END: Type Check

        accum_loss = 0 if train else None
        if train:
            if self.__gpu >= 0:
                y_batch = cuda.to_gpu(y_batch.data)
            accum_loss = F.softmax_cross_entropy(y, y_batch)

        return accum_loss

class MLP(ChainList):
    def __init__(self, dimensions,
                 prefix="MLP",
                 keep_rate=0.5,
                 gpu=-1,
                 ):
        super(MLP, self).__init__()
        self.keep_rate = keep_rate
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.layers = []

        assert len(dimensions) >= 2, "Must initialize MLP with 2 or more layers."
        for l0_dim, l1_dim in zip(dimensions[:-1], dimensions[1:]):
            self.add_link(L.Linear(l0_dim, l1_dim))

    def __call__(self, x_batch, train=True):
        ratio = 1 - self.keep_rate
        layers = self.layers
        h = x_batch

        for l0 in self.children():
            h = F.dropout(h, ratio, train)
            h = F.relu(l0(h))
        y = h
        return y


class BaseSentencePairTrainer(object):

    def __init__(self, model, gpu=-1, **kwargs):
        self.__gpu = gpu
        self.init_model(model)
        self.init_params()
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.model.to_gpu()

    def init_model(self, model):
        self.model = model
        self.model.add_persistent('best_dev_error', 0.0)
        self.model.add_persistent('step', 0)

    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        self.optimizer = optimizers.SGD(lr=lr)
        self.optimizer.setup(self.model)

    def update(self):
        self.optimizer.update()

    def forward(self, run_args, x_batch, y_batch=None, train=True, predict=False):
        assert "sentences" in x_batch and "transitions" in x_batch, \
            "Input must contain dictionary of sentences and transitions."

        sentences = x_batch["sentences"]
        transitions = x_batch["transitions"]

        ret = self.model(run_args, sentences, transitions, y_batch, train=train)
        y = ret[0]
        if predict:
            preds = self.__mod.argmax(y.data, 1).tolist()
        else:
            preds = None
        return ret

    def save(self, filename, step, best_dev_error):
        self.model.step = step
        self.model.best_dev_error = best_dev_error
        chainer.serializers.save_npz(filename, self.model)

    def load(self, filename):
        chainer.serializers.load_npz(filename, self.model)
        return self.model.step, self.model.best_dev_error


class Embed(Chain):
    """Flexible word embedding module for SPINN.

    It supports either word vectors trained from scratch or word vectors
    defined by a trainable projection layer on top of fixed embeddings.

    Args:
        size: The size of the model state; word vectors are treated as LSTM
            states with both ``c`` and ``h`` so their dimension is ``2*size``.
        vocab_size: The size of the vocabulary.
        dropout: The drop fraction for dropout on the word vectors.
        vectors: None, or an ~numpy.ndarray containing the fixed embeddings.
        normalization: A normalization link class (typically
            ~chainer.links.BatchNormalization).
    """

    def __init__(self, size, vocab_size, dropout, vectors,
                 make_buffers=True, activation=None,
                 use_input_dropout=False, use_input_norm=False):
        size = 2 * size if make_buffers else size
        if vectors is None:
            super(Embed, self).__init__(embed=L.EmbedID(vocab_size, size))
        else:
            super(Embed, self).__init__(projection=L.Linear(vectors.shape[1], size))
        if use_input_norm:
            self.add_link('normalization', L.BatchNormalization(size))
        self.vectors = vectors
        self.dropout = dropout
        self.make_buffers = make_buffers
        self.activation = (lambda x: x) if activation is None else activation
        self.use_input_dropout = use_input_dropout
        self.use_input_norm = use_input_norm

    def __call__(self, tokens):
        """Embed a tensor of tokens into a list of SPINN buffers.

        Returns the embeddings as a list of lists of ~chainer.Variable objects
        each augmented with ``tokens`` and ``transitions`` attributes, where
        each buffer (list of variables) is reversed to provide queue-like
        ``pop()`` behavior.

        Args:
            tokens (~chainer.Variable): Tensor of token ids with shape
                ``(B,T)``

        Returns:
            buffers: List of ``B`` lists, each of which contains ``T``
                variables in reverse timestep order, each of which contains
                the embedding of the corresponding token along with ``tokens``
                and ``transitions`` attributes.
        """
        b, l = tokens.data.shape[:2]
        if self.vectors is None:
            embeds = self.embed(to_gpu(F.reshape(tokens, (-1,))))
        else:
            embeds = Variable(
                arr_to_gpu(
                    self.vectors.take(tokens.data.ravel(), axis=0)),
                volatile=tokens.volatile)
            embeds = self.projection(embeds)
        if not self.make_buffers:
            return self.activation(F.reshape(embeds, (b, l, -1)))

        if self.use_input_norm:
            embeds = self.normalization(embeds, embeds.volatile == 'on')

        if self.use_input_dropout:
            embeds = F.dropout(embeds, self.dropout, embeds.volatile == 'off')

        return to_cpu(embeds)


class Reduce(Chain):
    """TreeLSTM composition module for SPINN.

    The TreeLSTM has two to four inputs: the first two are the left and right
    children being composed; the third is the current state of the tracker
    LSTM if one is present in the SPINN model; the fourth is an optional
    attentional input.

    Args:
        size: The size of the model state.
        tracker_size: The size of the tracker LSTM hidden state, or None if no
            tracker is present.
        attend (bool): Whether to accept an additional attention input.
        attn_fn (function): A callback function to compute the attention value
            given the left, right, tracker, and attention inputs. TODO
    """

    def __init__(self, size, tracker_size=None, attend=False, attn_fn=None):
        super(Reduce, self).__init__(
            left=L.Linear(size, 5 * size),
            right=L.Linear(size, 5 * size, nobias=True))
        if tracker_size is not None:
            self.add_link('track',
                          L.Linear(tracker_size, 5 * size, nobias=True))
        if attend:
            self.add_link('attend', L.Linear(size, 5 * size, nobias=True))
        self.attn_fn = lambda l, r, t, a: a if attn_fn is None else attn_fn

    def __call__(self, left_in, right_in, tracking=None, attend=None):
        """Perform batched TreeLSTM composition.

        This implements the REDUCE operation of a SPINN in parallel for a
        batch of nodes. The batch size is flexible; only provide this function
        the nodes that actually need to be REDUCEd.

        The TreeLSTM has two to four inputs: the first two are the left and
        right children being composed; the third is the current state of the
        tracker LSTM if one is present in the SPINN model; the fourth is an
        optional attentional input. All are provided as iterables and batched
        internally into tensors.

        Additionally augments each new node with pointers to its children as
        well as concatenated attributes ``transitions`` and ``tokens`` and
        the states of the buffer, stack, and tracker prior to the first SHIFT
        of a token that is part of the node. This allows restarting the
        encoding process with modifications to a particular node.

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
            attend: Iterable of ``B`` ~chainer.Variable objects containing
                ``c`` and ``h`` concatenated for the attention state to be fed
                into each node in the batch, or None.

        Returns:
            out: Tuple of ``B`` ~chainer.Variable objects containing ``c`` and
                ``h`` concatenated for the LSTM state of each new node. These
                objects are also augmented with ``left``, ``right``,
                ``tokens``, ``transitions``, ``buf``, ``stack``, and
                ``tracking`` attributes.
        """
        left, right = bundle(left_in), bundle(right_in)
        tracking, attend = bundle(tracking), bundle(attend)
        lstm_in = self.left(left.h)
        lstm_in += self.right(right.h)
        if hasattr(self, 'track'):
            lstm_in += self.track(tracking.h)
        if hasattr(self, 'attend'):
            lstm_in += self.attend(attend.h)
        out = unbundle(treelstm(left.c, right.c, lstm_in))
        for o, l, r in zip(out, left_in, right_in):
            if hasattr(l, 'buf'):
                o.left, o.right = l, r
                o.buf = o.left.buf
                # o.left.parent, o.right.parent = o, o
                o.transitions = o.left.transitions + o.right.transitions + [1]
                o.tokens = o.left.tokens + o.right.tokens
                o.stack = o.left.stack
                o.tracking = o.left.tracking
        return out
