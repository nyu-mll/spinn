from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

# Chainer imports
import chainer
from chainer import reporter, initializers
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
from chainer.functions.connection import embed_id
from chainer.functions.normalization.batch_normalization import batch_normalization
from chainer.functions.evaluation import accuracy
import chainer.links as L
from chainer.training import extensions

from chainer.functions.activation import slstm
from chainer.utils import type_check

from spinn.util.chainer_blocks import BaseSentencePairTrainer, Reduce
from spinn.util.chainer_blocks import LSTMState, Embed
from spinn.util.chainer_blocks import MLP
from spinn.util.chainer_blocks import CrossEntropyClassifier
from spinn.util.chainer_blocks import bundle, unbundle, the_gpu, to_cpu, to_gpu, treelstm

"""
Style Guide:

1. Each __call__() or forward() should be documented with its
   input and output types/dimensions.
2. Every ChainList/Chain/Link needs to have assigned a __gpu and __mod.
3. Each __call__() or forward() should have `train` as a parameter,
   and Variables need to be set to Volatile=True during evaluation.
4. Each __call__() or forward() should have an accompanying `check_type_forward`
   called along the lines of:

   ```
   in_data = tuple([x.data for x in [input_1, input_2]])
   in_types = type_check.get_types(in_data, 'in_types', False)
   self.check_type_forward(in_types)
   ```

   This is mimicing the behavior seen in Chainer Functions.
5. Each __call__() or forward() should have a chainer.Variable as input.
   There may be slight exceptions to this rule, since at a times
   especially in this model a list is preferred, but try to stick to
   this as close as possible. When avoiding this rule, consider setting
   a property rather than passing the variable. For instance:

   ```
   link.transitions = transitions
   loss = link(sentences)
   ```
6. Each link should be made to run on GPU and CPU.
7. Type checking should be disabled using an environment variable.

"""

T_SKIP   = 2
T_SHIFT  = 0
T_REDUCE = 1

def HeKaimingInit(shape, real_shape=None):
    # Calculate fan-in / fan-out using real shape if given as override
    fan = real_shape or shape

    return np.random.normal(scale=np.sqrt(4.0/(fan[0] + fan[1])),
                            size=shape)


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            if len(data.shape) >= 2:
                data[:] = HeKaimingInit(data.shape)
            else:
                data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, **kwargs):
        self.optimizer = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        # self.optimizer = optimizers.SGD(lr=0.01)
        self.optimizer.setup(self.model)
        # self.optimizer.add_hook(chainer.optimizer.GradientClipping(40))
        # self.optimizer.add_hook(chainer.optimizer.WeightDecay(0.00003))


class SentenceTrainer(SentencePairTrainer):
    pass

class Tracker(Chain):

    def __init__(self, size, tracker_size, predict, predict_skips=True, use_tracker_dropout=True, tracker_dropout_rate=0.1):
        super(Tracker, self).__init__(
            lateral=L.Linear(tracker_size, 4 * tracker_size),
            buf=L.Linear(size, 4 * tracker_size, nobias=True),
            stack1=L.Linear(size, 4 * tracker_size, nobias=True),
            stack2=L.Linear(size, 4 * tracker_size, nobias=True))
        if predict:
            self.add_link('transition', L.Linear(tracker_size, 2 + predict_skips))
        self.state_size = tracker_size
        self.tracker_dropout_rate = tracker_dropout_rate
        self.use_tracker_dropout = use_tracker_dropout
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, bufs, stacks):
        self.batch_size = len(bufs)
        zeros = Variable(np.zeros(bufs[0][0].shape, dtype=bufs[0][0].data.dtype),
                         volatile='auto')
        buf = bundle(buf[-1] for buf in bufs)
        stack1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in stacks)
        stack2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in stacks)

        lstm_in = self.buf(buf.h)
        lstm_in += self.stack1(stack1.h)
        lstm_in += self.stack2(stack2.h)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            self.c = Variable(
                self.xp.zeros((self.batch_size, self.state_size),
                              dtype=lstm_in.data.dtype),
                volatile='auto')

        if self.use_tracker_dropout:
            lstm_in = F.dropout(lstm_in, self.tracker_dropout_rate, train=lstm_in.volatile == False)

        self.c, self.h = F.lstm(self.c, lstm_in)
        if hasattr(self, 'transition'):
            return self.transition(self.h)
        return None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(Chain):

    def __init__(self, args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None, use_reinforce=True):
        super(SPINN, self).__init__(
            embed=Embed(args.size, vocab.size, args.input_dropout_rate,
                        vectors=vocab.vectors, normalization=normalization,
                        use_input_dropout=args.use_input_dropout,
                        use_input_norm=args.use_input_norm,
                        ),
            reduce=Reduce(args.size, args.tracker_size, attention, attn_fn))
        if args.tracker_size is not None:
            self.add_link('tracker', Tracker(
                args.size, args.tracker_size,
                predict=args.transition_weight is not None,
                predict_skips=False,
                use_tracker_dropout=args.use_tracker_dropout,
                tracker_dropout_rate=args.tracker_dropout_rate))
        self.transition_weight = args.transition_weight
        self.use_history = args.use_history
        self.save_stack = args.save_stack
        self.use_reinforce = use_reinforce

    def __call__(self, example, attention=None, print_transitions=False):
        self.tokens = example.tokens
        self.bufs = self.embed(example.tokens)
        # prepend with NULL NULL:
        # This exists specifically for the tracker.
        self.stacks = [[buf[0], buf[0]] for buf in self.bufs]
        for stack, buf in zip(self.stacks, self.bufs):
            for ss in stack:
                if self.save_stack:
                    ss.buf = buf[:]
                    ss.stack = stack[:]
                    ss.tracking = None
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if hasattr(example, 'transitions'):
            self.transitions = example.transitions
        self.attention = attention
        return self.run(run_internal_parser=True)

    def run(self, print_transitions=False, run_internal_parser=False,
            use_internal_parser=True):
        # how to use:
        # encoder.bufs = bufs, unbundled
        # encoder.stacks = stacks, unbundled
        # encoder.tracker.state = trackings, unbundled
        # encoder.transitions = ExampleList of Examples, padded with n
        # encoder.run()

        num_pad = [np.sum(seq == 0) for seq in self.tokens.data]
        self.history = [[] for buf in self.bufs] if self.use_history is not None \
                        else itertools.repeat(None)

        transition_loss, transition_acc = 0, 0
        if hasattr(self, 'transitions'):
            num_transitions = self.transitions.shape[1]
        else:
            num_transitions = len(self.bufs[0]) * 2 - 3

        for i in range(num_transitions):
            if hasattr(self, 'transitions'):
                transitions = self.transitions[:, i]
                transition_arr = list(transitions)
            else:
            #     transition_arr = [0]*len(self.bufs)
                raise Exception('Running without transitions not implemented')

            must_notskip = np.array([self.tokens.data[i][-len(buf)] != 0 for i, buf in enumerate(self.bufs)])
            if hasattr(self, 'tracker') and sum(must_notskip) > 0:
                buf_noskip = [buf if must_notskip[i] else Variable(np.zeros_like(buf.data)) for i, buf in enumerate(self.bufs) ]
                stack_noskip = [stack if must_notskip[i] else Variable(np.zeros_like(stack.data)) for i, stack in enumerate(self.stacks)]
                transition_hyp = self.tracker(buf_noskip, stack_noskip)

                if run_internal_parser:
                    transition_hyp = to_cpu(transition_hyp)
                    if self.use_reinforce:
                        probas = F.softmax(transition_hyp)
                        transition_preds = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                        transition_preds[must_notskip] = np.array([np.random.choice(2, 1, p=proba)[0] for proba in probas.data[must_notskip]])

                        validate_transitions = True
                        if validate_transitions:
                            # TODO: Almost definitely these don't work as expected because of how
                            # things are initialized and because of the SKIP action.

                            # Cannot reduce on too small a stack
                            must_shift = np.array([len(stack) < 2 for stack in self.stacks])
                            transition_preds[np.logical_and(must_shift, must_notskip)] = T_SHIFT

                            # Cannot shift if stack has to be reduced
                            must_reduce = np.array([len(buf) >= num_transitions - num_pad[i]  for i, buf in enumerate(self.bufs)])
                            transition_preds[np.logical_and(must_reduce, must_notskip)] = T_REDUCE


                        must_notskip =  np.tile(np.expand_dims(must_notskip, axis=1), (1, 2))

                        transition_acc += F.accuracy(probas, transitions)

                        # Probas are replaced by 50-50 on "SKIP" examples,
                        # but it doesn't get backprop thru anyway because
                        # the corresponding mask is 0 in F.where()
                        transition_loss += F.softmax_cross_entropy(
                            F.where(must_notskip, probas, Variable(0.5*np.ones_like(probas.data))),
                            np.where(must_notskip[:,0], transition_preds, T_SHIFT*np.ones_like(transition_preds)),
                            normalize=False)

                    else:
                        transition_preds = [T_SKIP for _ in range(len(self.bufs))]
                        transition_preds[must_notskip] = transition_hyp[must_notskip].data.argmax(axis=1)

                        transition_loss += F.softmax_cross_entropy(
                            transition_hyp, transitions,
                            normalize=False)
                        transition_acc += F.accuracy(transition_hyp, transitions)

                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

            lefts, rights, trackings, attentions = [], [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks, self.history,
                        self.tracker.states if hasattr(self, 'tracker')
                        else itertools.repeat(None),
                        self.attention if self.attention is not None
                        else itertools.repeat(None))

            assert len(transition_arr) == len(self.bufs)
            assert len(self.stacks) == len(self.bufs)

            for transition, buf, stack, history, tracking, attention in batch:
                must_shift = len(stack) < 2

                if transition == T_SHIFT: # shift
                    if self.save_stack:
                        buf[-1].buf = buf[:]
                        buf[-1].stack = stack[:]
                        buf[-1].tracking = tracking
                    stack.append(buf.pop())
                    if self.use_history:
                        history.append(stack[-1])
                elif transition == T_REDUCE: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            zeros = Variable(np.zeros(buf[0].shape,
                                dtype=buf[0].data.dtype),
                                volatile='auto')
                            if self.save_stack:
                                zeros.buf = buf[:]
                                zeros.stack = stack[:]
                                zeros.tracking = tracking
                            lr.append(zeros)
                    trackings.append(tracking)
                    attentions.append(attention)
                else:
                    if self.use_history:
                        history.append(buf[-1])  # pad history so it can be stacked/transposed

            if len(rights) > 0:
                reduced = iter(self.reduce(
                    lefts, rights, trackings, attentions))
                for transition, stack, history in zip(
                        transition_arr, self.stacks, self.history):
                    if transition == T_REDUCE: # reduce
                        new_stack_item = next(reduced)
                        assert isinstance(new_stack_item.data, np.ndarray), "Pushing cupy array to stack"
                        stack.append(new_stack_item)
                        if self.use_history:
                            history.append(stack[-1])
        if print_transitions:
            print()
        if self.transition_weight is not None and transition_loss is not 0:
            reporter.report({'transition_accuracy': transition_acc / num_transitions,
                             'transition_loss': transition_loss / num_transitions}, self)
            transition_loss *= self.transition_weight
        else:
            transition_loss = None

        return [stack.pop() for stack in self.stacks], transition_loss


class BaseModel(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 input_keep_rate, classifier_keep_rate,
                 use_tracker_dropout=True, tracker_dropout_rate=0.1,
                 use_input_dropout=False, use_input_norm=False,
                 use_classifier_norm=True,
                 gpu=-1,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_shift_composition=True,
                 make_logits=False,
                 use_history=False,
                 save_stack=False,
                 use_reinforce=False,
                 use_sentence_pair=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        the_gpu.gpu = gpu

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim
        self.add_link('l0', L.Linear(mlp_input_dim, mlp_dim))
        self.add_link('l1', L.Linear(mlp_dim, mlp_dim))
        self.add_link('l2', L.Linear(mlp_dim, num_classes))

        if use_classifier_norm:
            self.add_link('bn_0', L.BatchNormalization(mlp_input_dim))
            self.add_link('bn_1', L.BatchNormalization(mlp_dim))
            self.add_link('bn_2', L.BatchNormalization(mlp_dim))

        self.classifier = CrossEntropyClassifier(gpu)
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.accFun = accuracy.accuracy
        self.initial_embeddings = initial_embeddings
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.use_classifier_norm = use_classifier_norm
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.use_reinforce = use_reinforce

        args = {
            'size': model_dim/2,
            'tracker_size': tracking_lstm_hidden_dim if use_tracking_lstm else None,
            'transition_weight': transition_weight,
            'use_history': use_history,
            'save_stack': save_stack,
            'input_dropout_rate': 1. - input_keep_rate,
            'use_input_dropout': use_input_dropout,
            'use_input_norm': use_input_norm,
            'use_tracker_dropout': use_tracker_dropout,
            'tracker_dropout_rate': tracker_dropout_rate,
        }
        args = argparse.Namespace(**args)

        vocab = {
            'size': initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size,
            'vectors': initial_embeddings,
        }
        vocab = argparse.Namespace(**vocab)

        self.add_link('spinn', SPINN(args, vocab, normalization=L.BatchNormalization,
                 attention=False, attn_fn=None, use_reinforce=use_reinforce))


    def build_example(self, sentences, transitions, train):
        raise Exception('Not implemented.')


    def run_spinn(self, example, train):
        r = reporter.Reporter()
        r.add_observer('spinn', self.spinn)
        observation = {}
        with r.scope(observation):
            h_both, _ = self.spinn(example)

        transition_acc = observation.get('spinn/transition_accuracy', 0.0)
        transition_loss = observation.get('spinn/transition_loss', None)
        return h_both, transition_acc, transition_loss


    def run_mlp(self, h, train):
        # Pass through MLP Classifier.
        h = to_gpu(h)
        if hasattr(self, 'bn_0'):
            h = self.bn_0(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l0(h)
        h = F.relu(h)
        if hasattr(self, 'bn_1'):
            h = self.bn_1(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l1(h)
        h = F.relu(h)
        if hasattr(self, 'bn_2'):
            h = self.bn_2(h, not train)
        h = F.dropout(h, self.classifier_dropout_rate, train)
        h = self.l2(h)
        y = h

        return y


    def __call__(self, sentences, transitions, y_batch=None, train=True):
        example = self.build_example(sentences, transitions, train)
        h, transition_acc, transition_loss = self.run_spinn(example, train)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
        self.accuracy = self.accFun(y, self.__mod.array(y_batch))

        if hasattr(transition_acc, 'data'):
          transition_acc = transition_acc.data

        return y, accum_loss, self.accuracy.data, transition_acc, transition_loss

class SentencePairModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        assert batch_size * 2 == x.shape[0]
        assert batch_size * 2 == t.shape[0]

        example = {
            'tokens': Variable(x, volatile=not train),
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train):
        h_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(example, train)
        h_premise = F.concat(h_both[:batch_size], axis=0)
        h_hypothesis = F.concat(h_both[batch_size:], axis=0)
        h = F.concat([h_premise, h_hypothesis], axis=1)
        return h, transition_acc, transition_loss


class SentenceModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = {
            'tokens': Variable(x, volatile=not train),
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train):
        h, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(example, train)
        h = F.concat(h, axis=0)
        return h, transition_acc, transition_loss
