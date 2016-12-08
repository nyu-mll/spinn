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

    def __init__(self, size, tracker_size, predict, use_tracker_dropout=True, tracker_dropout_rate=0.1, use_skips=False):
        super(Tracker, self).__init__(
            lateral=L.Linear(tracker_size, 4 * tracker_size),
            buf=L.Linear(size, 4 * tracker_size, nobias=True),
            stack1=L.Linear(size, 4 * tracker_size, nobias=True),
            stack2=L.Linear(size, 4 * tracker_size, nobias=True))
        if predict:
            self.add_link('transition', L.Linear(tracker_size, 3 if use_skips else 2))
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

    def __init__(self, args,
                 attention=False, attn_fn=None, use_reinforce=True, use_skips=False):
        super(SPINN, self).__init__(
            reduce=Reduce(args.projection_dim, args.tracker_size, attention, attn_fn))
        if args.tracker_size is not None:
            self.add_link('tracker', Tracker(
                args.projection_dim, args.tracker_size,
                predict=args.transition_weight is not None,
                use_tracker_dropout=args.use_tracker_dropout,
                tracker_dropout_rate=args.tracker_dropout_rate, use_skips=use_skips))
        self.transition_weight = args.transition_weight
        self.use_history = args.use_history
        self.save_stack = args.save_stack
        self.use_reinforce = use_reinforce
        self.use_skips = use_skips
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def __call__(self, run_args, example):
        self.bufs = example.tokens
        self.stacks = [[] for buf in self.bufs]
        self.buffers_t = [0 for buf in self.bufs]
        # There are 2 * N - 1 transitons, so (|transitions| + 1) / 2 should equal N.
        self.buffers_n = [(len([t for t in ts if t != T_SKIP]) + 1) / 2 for ts in example.transitions]
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
        return self.run(run_args)

    def validate(self, transitions, preds, stacks, buffers_t, buffers_n):
        # TODO: Almost definitely these don't work as expected because of how
        # things are initialized and because of the SKIP action.

        DEFAULT_CHOICE = T_SHIFT
        cant_skip = np.array([p == T_SKIP and t != T_SKIP for t, p in zip(transitions, preds)])
        preds[cant_skip] = DEFAULT_CHOICE

        # Cannot reduce on too small a stack
        must_shift = np.array([len(stack) < 2 for stack in stacks])
        preds[must_shift] = T_SHIFT

        # Cannot shift if stack has to be reduced
        must_reduce = np.array([buf_t >= buf_n for buf_t, buf_n in zip(buffers_t, buffers_n)])
        preds[must_reduce] = T_REDUCE

        must_skip = np.array([t == T_SKIP for t in transitions])
        preds[must_skip] = T_SKIP

        return preds

    def run(self, run_args):
        # how to use:
        # encoder.bufs = bufs, unbundled
        # encoder.stacks = stacks, unbundled
        # encoder.tracker.state = trackings, unbundled
        # encoder.transitions = ExampleList of Examples, padded with n
        # encoder.run()
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

            cant_skip = np.array([t != T_SKIP for t in transitions])
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):
                transition_hyp = self.tracker(self.bufs, self.stacks)
                if transition_hyp is not None and run_args.run_internal_parser:
                    transition_hyp = to_cpu(transition_hyp)
                    if hasattr(self, 'transitions'):
                        memory = {}
                        if self.use_reinforce:
                            probas = F.softmax(transition_hyp)
                            samples = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                            samples[cant_skip] = [np.random.choice(self.choices, 1, p=proba)[0] for proba in probas.data[cant_skip]]

                            transition_preds = samples
                            hyp_acc = probas
                            hyp_xent = probas
                            truth_acc = transitions
                            truth_xent = samples
                        else:
                            transition_preds = transition_hyp.data.argmax(axis=1)
                            hyp_acc = transition_hyp
                            hyp_xent = transition_hyp
                            truth_acc = transitions
                            truth_xent = transitions

                        if run_args.use_random:
                            print("Using random")
                            transition_preds = np.random.choice(self.choices, len(self.bufs))
                        
                        if run_args.validate_transitions:
                            transition_preds = self.validate(transition_arr, transition_preds,
                                self.stacks, self.buffers_t, self.buffers_n)

                        memory["logits"] = transition_hyp
                        memory["preds"]  = transition_preds

                        if not self.use_skips:
                            hyp_acc = hyp_acc.data[cant_skip]
                            truth_acc = truth_acc[cant_skip]

                            cant_skip_mask = np.tile(np.expand_dims(cant_skip, axis=1), (1, 2))
                            hyp_xent = F.split_axis(transition_hyp, transition_hyp.shape[0], axis=0)
                            hyp_xent = F.concat([hyp_xent[i] for i, y in enumerate(cant_skip) if y], axis=0)
                            truth_xent = truth_xent[cant_skip]

                        memory["hyp_acc"] = hyp_acc
                        memory["truth_acc"] = truth_acc
                        memory["hyp_xent"] = hyp_xent
                        memory["truth_xent"] = truth_xent

                        memory["preds_cm"] = np.array(transition_preds[cant_skip])
                        memory["truth_cm"] = np.array(transitions[cant_skip])

                        if run_args.use_internal_parser:
                            transition_arr = transition_preds.tolist()

                        self.memories.append(memory)

            lefts, rights, trackings, attentions = [], [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks, self.history,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None),
                        itertools.repeat(None))

            for ii, (transition, buf, stack, history, tracking, attention) in enumerate(batch):
                must_shift = len(stack) < 2

                if transition == T_SHIFT: # shift
                    if self.save_stack:
                        buf[-1].buf = buf[:]
                        buf[-1].stack = stack[:]
                        buf[-1].tracking = tracking
                    stack.append(buf.pop())
                    self.buffers_t[ii] += 1
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
        if run_args.print_transitions:
            print()
        if self.transition_weight is not None:
            # We compute statistics after the fact, since sub-batches can
            # have different sizes when not using skips.
            statistics = zip(*[
                (m["hyp_acc"], m["truth_acc"], m["hyp_xent"], m["truth_xent"])
                for m in self.memories])

            statistics = [
                F.squeeze(F.concat([F.expand_dims(ss, 1) for ss in s], axis=0))
                if isinstance(s[0], Variable) else
                np.array(reduce(lambda x, y: x + y.tolist(), s, []))
                for s in statistics]

            hyp_acc, truth_acc, hyp_xent, truth_xent = statistics

            transition_acc = F.accuracy(
                hyp_acc, truth_acc.astype(np.int32))
            transition_loss = F.softmax_cross_entropy(
                hyp_xent, truth_xent.astype(np.int32),
                normalize=False)

            reporter.report({'transition_accuracy': transition_acc,
                             'transition_loss': transition_loss}, self)
            transition_loss *= self.transition_weight
        else:
            transition_loss = None

        return [stack[-1] for stack in self.stacks], transition_loss


class LSTMChain(Chain):
    def __init__(self, input_dim, hidden_dim, seq_length, gpu=-1):
        super(LSTMChain, self).__init__(
            i_fwd=L.Linear(input_dim, 4 * hidden_dim, nobias=True),
            h_fwd=L.Linear(hidden_dim, 4 * hidden_dim),
        )
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np

    def __call__(self, x_batch, train=True, keep_hs=False, reverse=False):
        batch_size = x_batch.data.shape[0]
        c = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        h = self.__mod.zeros((batch_size, self.hidden_dim), dtype=self.__mod.float32)
        hs = []
        batches = F.split_axis(x_batch, self.seq_length, axis=1)
        if reverse:
            batches = list(reversed(batches))
        for x in batches:
            ii = self.i_fwd(x)
            hh = self.h_fwd(h)
            ih = ii + hh
            c, h = F.lstm(c, ih)

            if keep_hs:
                # Convert from (#batch_size, #hidden_dim) ->
                #              (#batch_size, 1, #hidden_dim)
                # This is important for concatenation later.
                h_reshaped = F.reshape(h, (batch_size, 1, self.hidden_dim))
                hs.append(h_reshaped)

        if keep_hs:
            # This converts list of: [(#batch_size, 1, #hidden_dim)]
            # To single tensor:       (#batch_size, #seq_length, #hidden_dim)
            # Which matches the input shape.
            if reverse:
                hs = list(reversed(hs))
            hs = F.concat(hs, axis=1)
        else:
            hs = None

        return c, h, hs


class BaseModel(Chain):
    def __init__(self, args, temp_args):
        super(BaseModel, self).__init__()

        # Extract args
        
        self.args = args
        mlp_input_dim = args.model_dim * 2 if args.use_sentence_pair else args.model_dim
        
        self.classifier = CrossEntropyClassifier(args.gpu)
        self.__mod = cuda.cupy if args.gpu >= 0 else np
        self.accFun = accuracy.accuracy
        
        the_gpu.gpu = args.gpu

        self.add_link('l0', L.Linear(mlp_input_dim, args.mlp_dim))
        self.add_link('l1', L.Linear(args.mlp_dim, args.mlp_dim))
        self.add_link('l2', L.Linear(args.mlp_dim, args.num_classes))

        self.add_link('embed', 
                    Embed(args.projection_dim, args.vocab_size, args.input_dropout_rate,
                        vectors=temp_args.initial_embeddings,
                        use_input_dropout=args.use_input_dropout,
                        use_input_norm=args.use_input_norm,
                        ))

        self.add_link('spinn', SPINN(args,
                 attention=False, attn_fn=None, use_reinforce=args.use_reinforce, use_skips=args.use_skips))

        if args.use_encode:
            # TODO: Could probably have a buffer that is [concat(embed, fwd, bwd)] rather
            # than just [concat(fwd, bwd)]. More generally, [concat(embed, activation(embed))].
            self.add_link('fwd_rnn', LSTMChain(input_dim=args.projection_dim * 2, hidden_dim=args.model_dim/2, seq_length=args.seq_length))
            self.add_link('bwd_rnn', LSTMChain(input_dim=args.projection_dim * 2, hidden_dim=args.model_dim/2, seq_length=args.seq_length))


    def build_example(self, sentences, transitions, train):
        raise Exception('Not implemented.')


    def run_embed(self, example, train):
        embeds = self.embed(example.tokens)

        b, l = example.tokens.shape[:2]

        embeds = F.split_axis(to_cpu(embeds), b, axis=0, force_tuple=True)
        embeds = [F.expand_dims(x, 0) for x in embeds]
        embeds = F.concat(embeds, axis=0)

        if self.args.use_encode:
            _, _, fwd_hs = self.fwd_rnn(embeds, train, keep_hs=True)
            _, _, bwd_hs = self.bwd_rnn(embeds, train, keep_hs=True, reverse=True)
            hs = F.concat([fwd_hs, bwd_hs], axis=2)
            embeds = hs

        embeds = [F.split_axis(x, l, axis=0, force_tuple=True) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        assert b == len(buffers)

        example.tokens = buffers

        return example


    def run_spinn(self, run_args, example, train):
        r = reporter.Reporter()
        r.add_observer('spinn', self.spinn)
        observation = {}
        with r.scope(observation):
            self.spinn.reset_state()
            h_both, _ = self.spinn(run_args, example)

        transition_acc = observation.get('spinn/transition_accuracy', 0.0)
        transition_loss = observation.get('spinn/transition_loss', None)
        return h_both, transition_acc, transition_loss


    def run_mlp(self, h, train):
        # Pass through MLP Classifier.
        h = to_gpu(h)
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = h

        return y


    def __call__(self, run_args, sentences, transitions, y_batch=None, train=True):
        example = self.build_example(sentences, transitions, train)
        example = self.run_embed(example, train)
        h, transition_acc, transition_loss = self.run_spinn(run_args, example, train)
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


    def run_spinn(self, run_args, example, train):
        h_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            run_args, example, train)
        batch_size = len(h_both) / 2
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


    def run_spinn(self, run_args, example, train):
        h, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            run_args, example, train)
        h = F.concat(h, axis=0)
        return h, transition_acc, transition_loss
