from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import LSTMState, Reduce
from spinn.util.blocks import bundle, unbundle, to_cuda
from spinn.util.blocks import treelstm, expand_along, dropout, expand_dims, select_item
from spinn.util.blocks import get_c, get_h, get_state
from spinn.util.blocks import BaseSentencePairTrainer
from spinn.util.blocks import Linear, LSTM, LSTMCell
from spinn.util.blocks import HeKaimingInit, ZeroInitializer

from sklearn import metrics

import spinn.cbow


T_SHIFT  = 0
T_REDUCE = 1
T_SKIP   = 2


"""
TODO:

- [x] Weight Initialization
    - [x] Projection
    - [x] Encoding
    - [x] Reduce
    - [x] Transition
    - [x] MLP
- [ ] Gradient Clipping
- [ ] Add GRU option for encoding layer and for tracker.

"""


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_optimizer(self, lr=0.01, l2_lambda=0.0, opt="RMSprop", **kwargs):
        relevant_params = [w for w in self.model.parameters() if w.requires_grad]
        if opt == "RMSprop":
            self.optimizer = optim.RMSprop(relevant_params, lr=lr, alpha=0.9, weight_decay=l2_lambda)
        elif opt == "Adam":
            self.optimizer = optim.Adam(relevant_params, lr=lr, betas=(0.9, 0.999), weight_decay=l2_lambda)
        else:
            raise NotImplementedError()


class SentenceTrainer(SentencePairTrainer):
    pass


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict, tracker_dropout_rate=0.0, use_skips=False):
        super(Tracker, self).__init__()
        self.buf = Linear(size, 4 * tracker_size, bias=False, initializer=HeKaimingInit)
        self.stack1 = Linear(size, 4 * tracker_size, bias=False, initializer=HeKaimingInit)
        self.stack2 = Linear(size, 4 * tracker_size, bias=False, initializer=HeKaimingInit)
        self.lstm = LSTMCell(4 * tracker_size, tracker_size, bias=False, initializer=HeKaimingInit)
        if predict:
            self.transition = Linear(tracker_size, 3 if use_skips else 2, initializer=HeKaimingInit)
        self.state_size = tracker_size
        self.tracker_dropout_rate = tracker_dropout_rate
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, bufs, stacks, train):
        self.batch_size = len(bufs)
        zeros = Variable(to_cuda(torch.from_numpy(
            np.zeros(bufs[0][0].size(), dtype=np.float32),
            ), self.gpu), volatile=not train)
        buf = bundle(buf[-1] for buf in bufs)
        stack1 = bundle(stack[-1] if len(stack) > 0 else zeros for stack in stacks)
        stack2 = bundle(stack[-2] if len(stack) > 1 else zeros for stack in stacks)

        lstm_in = self.buf(buf.h)
        lstm_in += self.stack1(stack1.h)
        lstm_in += self.stack2(stack2.h)
        if self.c is None:
            self.c = Variable(to_cuda(torch.from_numpy(
                np.zeros((self.batch_size, self.state_size),
                              dtype=np.float32)), self.gpu),
                volatile=not train)
        if self.h is None:
            self.h = Variable(to_cuda(torch.from_numpy(
                np.zeros((self.batch_size, self.state_size),
                              dtype=np.float32)), self.gpu),
                volatile=not train)

        # TODO: Tracker dropout.

        self.h, self.c = self.lstm(lstm_in, (self.h, self.c))
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


class SPINN(nn.Module):

    def __init__(self, args, vocab, use_skips=False):
        super(SPINN, self).__init__()
        self.reduce = Reduce(args.size, args.tracker_size, use_tracking_in_composition=args.use_tracking_in_composition)
        if args.tracker_size is not None:
            self.tracker = Tracker(
                args.size, args.tracker_size,
                predict=args.transition_weight is not None,
                tracker_dropout_rate=args.tracker_dropout_rate, use_skips=use_skips)
        self.transition_weight = args.transition_weight
        self.use_skips = use_skips
        choices = [T_SHIFT, T_REDUCE, T_SKIP] if use_skips else [T_SHIFT, T_REDUCE]
        self.choices = np.array(choices, dtype=np.int32)


    def forward(self, example, train, print_transitions=False, use_internal_parser=False,
                 validate_transitions=True,
                 use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        self.bufs = example.tokens
        self.stacks = [[] for buf in self.bufs]
        self.buffers_t = [0 for buf in self.bufs]
        self.memories = []
        self.transition_mask = np.zeros((len(example.tokens), len(example.tokens[0])), dtype=bool)

        # There are 2 * N - 1 transitons, so (|transitions| + 1) / 2 should equal N.
        self.buffers_n = [(len([t for t in ts if t != T_SKIP]) + 1) / 2 for ts in example.transitions]
        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if hasattr(example, 'transitions'):
            self.transitions = example.transitions
        return self.run(train, run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions,
                        use_reinforce=use_reinforce,
                        rl_style=rl_style,
                        rl_baseline=rl_baseline,
                        )

    def run(self, train, print_transitions=False, run_internal_parser=False, use_internal_parser=False,
            validate_transitions=True, use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
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
                raise NotImplementedError('Running without transitions not implemented.')

            cant_skip = np.array([t != T_SKIP for t in transitions])
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):
                transition_hyp = self.tracker(self.bufs, self.stacks, train)
                if transition_hyp is not None and run_internal_parser:
                    if hasattr(self, 'transitions'):
                        memory = {}
                        truth_acc = transitions
                        hyp_xent = transition_hyp
                        if use_reinforce:
                            probas = F.softmax(transition_hyp)
                            samples = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                            samples[cant_skip] = [np.random.choice(self.choices, 1, p=proba)[0] for proba in probas.data.cpu().numpy()[cant_skip]]

                            transition_preds = samples
                            hyp_acc = probas
                            truth_xent = samples
                        else:
                            transition_preds = transition_hyp.data.cpu().numpy().argmax(axis=1)
                            hyp_acc = transition_hyp
                            truth_xent = transitions

                        if validate_transitions:
                            transition_preds = self.validate(transition_arr, transition_preds,
                                self.stacks, self.buffers_t, self.buffers_n)

                        memory["logits"] = transition_hyp
                        memory["preds"]  = transition_preds

                        if not self.use_skips:
                            hyp_acc = hyp_acc.data.cpu().numpy()[cant_skip]
                            truth_acc = truth_acc[cant_skip]

                            cant_skip_mask = np.tile(np.expand_dims(cant_skip, axis=1), (1, 2))
                            hyp_xent = torch.chunk(transition_hyp, transition_hyp.size(0), 0)
                            hyp_xent = torch.cat([hyp_xent[iii] for iii, y in enumerate(cant_skip) if y], 0)
                            truth_xent = truth_xent[cant_skip]

                        self.transition_mask[cant_skip, i] = True

                        memory["hyp_acc"] = hyp_acc
                        memory["truth_acc"] = truth_acc
                        memory["hyp_xent"] = hyp_xent
                        memory["truth_xent"] = truth_xent

                        memory["preds_cm"] = np.array(transition_preds[cant_skip])
                        memory["truth_cm"] = np.array(transitions[cant_skip])

                        if use_internal_parser:
                            transition_arr = transition_preds.tolist()

                        self.memories.append(memory)

            lefts, rights, trackings = [], [], []
            batch = zip(transition_arr, self.bufs, self.stacks,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None))

            for ii, (transition, buf, stack, tracking) in enumerate(batch):
                must_shift = len(stack) < 2

                if transition == T_SHIFT: # shift
                    stack.append(buf.pop())
                    self.buffers_t[ii] += 1
                elif transition == T_REDUCE: # reduce
                    for lr in [rights, lefts]:
                        if len(stack) > 0:
                            lr.append(stack.pop())
                        else:
                            # TODO: This only happens in SNLI eval for some reason...
                            zeros = Variable(to_cuda(torch.from_numpy(np.zeros(buf[0].size(),
                                dtype=np.float32)), self.gpu),
                                volatile=not train)
                            lr.append(zeros)
                    trackings.append(tracking)
                else: # skip
                    pass
            if len(rights) > 0:
                reduced = iter(self.reduce(
                    lefts, rights, trackings))
                for transition, stack in zip(
                        transition_arr, self.stacks):
                    if transition == T_REDUCE: # reduce
                        new_stack_item = next(reduced)
                        stack.append(new_stack_item)

        if self.transition_weight is not None:
            # We compute statistics after the fact, since sub-batches can
            # have different sizes when not using skips.
            hyp_acc, truth_acc, hyp_xent, truth_xent = self.get_statistics()

            t_pred = hyp_acc.argmax(axis=1)
            transition_acc = np.asarray((t_pred == truth_acc).mean(dtype=np.float32))

            transition_logits = F.log_softmax(hyp_xent)
            transition_y = to_cuda(torch.from_numpy(truth_acc.astype(np.int32)).long(), self.gpu)
            transition_loss = F.nll_loss(
                transition_logits, Variable(transition_y, volatile=not train))

            transition_loss *= self.transition_weight
        else:
            transition_loss = None

        return [stack[-1] for stack in self.stacks], transition_loss, transition_acc


    def validate(self, transitions, preds, stacks, buffers_t, buffers_n):
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


    def get_statistics(self):
        statistics = zip(*[
            (m["hyp_acc"], m["truth_acc"], m["hyp_xent"], m["truth_xent"])
            for m in self.memories])

        statistics = [
            torch.squeeze(torch.cat([expand_dims(ss, 1) for ss in s], 0))
            if isinstance(s[0], Variable) else
            np.array(reduce(lambda x, y: x + y.tolist(), s, []))
            for s in statistics]

        hyp_acc, truth_acc, hyp_xent, truth_xent = statistics
        return hyp_acc, truth_acc, hyp_xent, truth_xent


class MLP(nn.Module):
    def __init__(self, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn,
                 classifier_dropout_rate=0.0):
        super(MLP, self).__init__()

        self.num_mlp_layers = num_mlp_layers
        self.mlp_bn = mlp_bn
        self.classifier_dropout_rate = classifier_dropout_rate

        features_dim = mlp_input_dim
        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear(features_dim, mlp_dim,
                initializer=HeKaimingInit))
            if mlp_bn:
                setattr(self, 'bn{}'.format(i), nn.BatchNorm1d(mlp_dim))
            features_dim = mlp_dim
        setattr(self, 'l{}'.format(num_mlp_layers), Linear(features_dim, num_classes,
                initializer=HeKaimingInit))

    def forward(self, h, train):
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if self.mlp_bn:
                bn = getattr(self, 'bn{}'.format(i))
                h = bn(h)
            h = dropout(h, self.classifier_dropout_rate, train)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        h = layer(h)
        y = F.log_softmax(h)
        return y


class BaseModel(nn.Module):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 embedding_keep_rate, classifier_keep_rate,
                 project_embeddings=True,
                 tracker_keep_rate=1.0,
                 use_input_norm=False,
                 gpu=-1,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_tracking_in_composition=True,
                 projection_dim=None,
                 encoding_dim=None,
                 use_encode=False,
                 use_skips=False,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 num_mlp_layers=2,
                 mlp_bn=False,
                 rl_baseline=None,
                 rl_policy_dim=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.model_dim = model_dim
        self.stack_hidden_dim = model_dim / 2
        mlp_input_dim = self.stack_hidden_dim * 2 if use_sentence_pair else self.stack_hidden_dim

        # Only enable vector comparison features for sentence_pair data.
        self.use_difference_feature = (use_difference_feature and use_sentence_pair)
        self.use_product_feature = (use_product_feature and use_sentence_pair)
        self.use_sentence_pair = use_sentence_pair

        # RL Params
        self.reinforce_lr = 0.01
        self.mu = 0.1
        self.baseline = 0

        if rl_baseline == "policy":
            raise NotImplementedError()

        self.initial_embeddings = initial_embeddings
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.embedding_dropout_rate = 1. - embedding_keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.use_encode = use_encode
        self.bi_encode = True
        self.transition_weight = transition_weight

        args = {
            'size': self.stack_hidden_dim,
            'tracker_size': tracking_lstm_hidden_dim if use_tracking_lstm else None,
            'use_tracking_in_composition': use_tracking_in_composition,
            'transition_weight': transition_weight,
            'use_input_norm': use_input_norm,
            'tracker_dropout_rate': 1. - tracker_keep_rate,
        }
        args = argparse.Namespace(**args)

        vocab = {
            'size': initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size,
            'vectors': initial_embeddings,
        }
        vocab = argparse.Namespace(**vocab)

        if initial_embeddings is not None:
            self._embed = nn.Embedding(vocab_size, word_embedding_dim)
            self._embed.weight.data.set_(torch.from_numpy(initial_embeddings))
            self._embed.weight.requires_grad = False
        else:
            self._embed = nn.Embedding(vocab_size, word_embedding_dim)
            self._embed.weight.requires_grad = True

        if project_embeddings and not self.use_encode:
            self.project = Linear(word_embedding_dim, model_dim, initializer=HeKaimingInit)
        else:
            self.project = lambda x: x

        if self.use_encode:
            raise NotImplementedError("Checkpoints are broken with use_encode.")
            bi = 2 if self.bi_encode else 1
            self.encode = LSTM(word_embedding_dim, model_dim / bi, 1,
                batch_first=True,
                bidirectional=self.bi_encode,
                initializer=HeKaimingInit,
                )

        self.spinn = SPINN(args, vocab, use_skips=use_skips)

        features_dim = mlp_input_dim
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.stack_hidden_dim
            if self.use_product_feature:
                features_dim += self.stack_hidden_dim

        self.mlp = MLP(features_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn,
            classifier_dropout_rate=self.classifier_dropout_rate)

        # TODO: Add HeKaimingInit in a reset_parameters() method.

        print(self)


    def build_example(self, sentences, transitions, train):
        raise NotImplementedError()


    def build_h(self, h):
        if self.use_sentence_pair:
            # Extract both example outputs, and strip off 'c' states.
            prem, hyp = h[:, :self.stack_hidden_dim], h[:, self.stack_hidden_dim*2:self.stack_hidden_dim*3]
            hs = [prem, hyp]
            if self.use_difference_feature:
                hs.append(prem - hyp)
            if self.use_product_feature:
                hs.append(prem * hyp)
            h = torch.cat(hs, 1)
        else:
            # Strip off 'c' states.
            h = h[:, :self.stack_hidden_dim]
        return h


    def run_embed(self, tokens, train):
        return self._embed(tokens)


    def run_encode(self, x, train):
        batch_size, seq_len, model_dim = x.size()

        num_layers = 1
        bidirectional = self.bi_encode
        bi = 2 if bidirectional else 1
        h0 = Variable(to_cuda(torch.zeros(num_layers * bi, batch_size, self.model_dim / bi), self.gpu), volatile=not train)
        c0 = Variable(to_cuda(torch.zeros(num_layers * bi, batch_size, self.model_dim / bi), self.gpu), volatile=not train)

        # Expects (input, h_0, c_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x bi[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.encode(x, (h0, c0))

        return output


    def build_buffers(self, emb, batch_size, seq_length):
        # Split twice:
        # 1. (#batch_size x #seq_length, #embed_dim) => [(#seq_length, #embed_dim)] x #batch_size
        # 2. (#seq_length, #embed_dim) => [(1, #embed_dim)] x #seq_length
        emb = [torch.chunk(x, seq_length, 0) for x in torch.chunk(emb, batch_size, 0)]
        buffers = [list(reversed(x)) for x in emb]
        return buffers


    def run_spinn(self, example, train, use_internal_parser,
                  validate_transitions=True,
                  use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h_both, transition_loss, transition_acc = self.spinn(example, train,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions,
                               use_reinforce=use_reinforce,
                               rl_style=rl_style,
                               rl_baseline=rl_baseline,
                               )
        return h_both, transition_acc, transition_loss


    def forward(self, sentences, transitions, y_batch=None, train=True,
                 use_reinforce=False, rl_style="zero-one", rl_baseline="ema",
                 use_internal_parser=False, validate_transitions=True):
        example = self.build_example(sentences, transitions, train)

        tokens = example.tokens

        emb = self.run_embed(tokens, train)
        emb = dropout(emb, self.embedding_dropout_rate, train)

        batch_size, seq_length, _ = emb.size()

        if self.use_encode:
            emb = self.run_encode(emb, train)
            # Flatten after.
            emb = emb.contiguous().view(-1, self.model_dim)
        else:
            # Flatten first.
            emb = emb.view(-1, self.word_embedding_dim)
            emb = self.project(emb)

        buffers = self.build_buffers(emb, batch_size, seq_length)

        example.tokens = buffers

        h, transition_acc, transition_loss = self.run_spinn(example, train, use_internal_parser,
            validate_transitions, use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
        h = self.build_h(h)
        y = self.mlp(h, train)

        # Calculate Loss & Accuracy.
        if y_batch is not None:
            loss = F.nll_loss(y, Variable(y_batch, volatile=not train))
            pred = y.data.max(1)[1] # get the index of the max log-probability
            acc = pred.eq(y_batch).sum() / float(y_batch.size(0))
        else:
            loss = None
            acc = 0.0

        if train and use_reinforce:
            rewards = self.build_rewards(y, y_batch, rl_style)

            if rl_baseline == "ema": # Exponential Moving Average
                self.baseline = self.baseline * (1 - self.mu) + self.mu * rewards.mean()
                self.avg_baseline = self.baseline
                new_rewards = rewards - self.baseline
            elif rl_baseline == "policy": # Policy Net
                baseline, baseline_loss = self.run_policy(sentences, transitions, y_batch, train, rewards, rl_style)
                self.avg_baseline = baseline.data.mean()
                new_rewards = rewards - baseline.data
            elif rl_baseline == "greedy": # Greedy Max
                baseline = self.run_greedy_max(sentences, transitions, y_batch, train, rewards, rl_style)
                self.avg_baseline = baseline.mean()
                new_rewards = rewards - baseline
            else:
                raise NotImplementedError()

            self.avg_reward = rewards.mean()
            self.avg_new_rew = new_rewards.mean()

            rl_loss = self.reinforce(new_rewards)
            rl_loss *= self.transition_weight
        else:
            rl_loss = None

        if hasattr(transition_acc, 'data'):
            transition_acc = transition_acc.data

        if rl_baseline == "policy" and baseline_loss is not None:
            rl_loss += baseline_loss

        return y, loss, acc, transition_acc, transition_loss, rl_loss


    def run_policy(self, sentences, transitions, y_batch, train, rewards, rl_style):
        if rl_style != "zero-one":
            raise NotImplementedError("Policy net is only compatible with zero-one loss right now."
                "It predicts a single value between 0 and 1.")
        y, _, _, _, _ = self.policy(sentences, transitions, y_batch=None, train=train)
        pred_reward = F.flatten(F.sigmoid(y)) # Squash between 0 and 1
        return pred_reward, F.mean_squared_error(pred_reward, rewards)


    def run_greedy_max(self, sentences, transitions, y_batch, train, rewards, rl_style):
        # TODO: Should this be run with train=False? Will effect batchnorm and dropout.
        y, _, _, _, _ = self.forward(sentences, transitions, y_batch=None, train=train, use_internal_parser=True)
        pred_reward = self.build_rewards(y, y_batch, rl_style)
        return pred_reward


    def build_rewards(self, logits, y, style="zero-one"):
        if style == "xent":
            rewards = -1. * F.concat([expand_dims(
                        F.softmax_cross_entropy(logits[i:(i+1)], y[i:(i+1)]), axis=0)
                        for i in range(y.shape[0])], axis=0).data
        elif style == "zero-one":
            rewards = torch.eq(logits.max(1)[1].data, y).float()
        else:
            raise NotImplementedError()
        return rewards


    def reinforce(self, rewards):
        """ The tricky step here is when we "expand rewards".

            Say we have batch size 2, with these actions, log_probs, and rewards:

            actions = [[0, 1], [1, 1]]
            log_probs = [
                [[0.2, 0.8], [0.3, 0.7]],
                [[0.4, 0.6], [0.5, 0.5]]
                ]
            rewards = [0., 1.]

            Then we want to calculate the objective as so:

            rl_loss = [0.2, 0.7, 0.6, 0.5] * [0., 0., 1., 1.]

            Now this gets slightly tricker when using skips (action==2):

            actions = [[0, 1], [2, 1]]
            log_probs = [
                [[0.2, 0.8], [0.3, 0.7]],
                [[0.4, 0.6], [0.5, 0.5]]
                ]
            rewards = [0., 1.]
            rl_loss = [0.2, 0.7, 0.5] * [0., 0., 1.]

            NOTE: The above example is fictional, and although those values are
            not achievable, is still representative of what is going on.

        """
        hyp_acc, truth_acc, hyp_xent, truth_xent = self.spinn.get_statistics()
        log_p = F.log_softmax(hyp_xent)
        log_p_preds = select_item(log_p, to_cuda(torch.from_numpy(truth_xent), self.gpu), self.gpu)

        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            rewards = torch.cat([rewards, rewards], 0)

        # Expand rewards
        if self.spinn.use_skips:
            rewards = expand_along(rewards.view(-1).cpu().numpy(), np.full(self.spinn.transition_mask.shape, True))
        else:
            rewards = expand_along(rewards.view(-1).cpu().numpy(), self.spinn.transition_mask)

        rl_loss = -1. * torch.dot(log_p_preds, Variable(to_cuda(torch.from_numpy(rewards), self.gpu), volatile=log_p_preds.volatile)) / log_p_preds.size(0)

        return rl_loss


class SentencePairModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.size(0)

        # Build Tokens
        x_prem = Variable(sentences[:,:,0], volatile=not train)
        x_hyp = Variable(sentences[:,:,1], volatile=not train)
        x = torch.cat([x_prem, x_hyp], 0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = {
            'tokens': x,
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True,
                  use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions,
            use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
        batch_size = len(h_both) / 2
        h_premise = torch.cat(h_both[:batch_size], 0)
        h_hypothesis = torch.cat(h_both[batch_size:], 0)
        h = torch.cat([h_premise, h_hypothesis], 1)
        return h, transition_acc, transition_loss


class SentenceModel(BaseModel):
    def build_example(self, sentences, transitions, train):
        batch_size = sentences.size(0)

        # Build Tokens
        x = Variable(sentences, volatile=not train)

        # Build Transitions
        t = transitions

        example = {
            'tokens': x,
            'transitions': t
        }
        example = argparse.Namespace(**example)

        return example


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True,
                  use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions,
            use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
        h = torch.cat(h, 0)
        return h, transition_acc, transition_loss
