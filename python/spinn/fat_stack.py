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

from spinn.util.blocks import LSTMState, Reduce, LSTMChain
from spinn.util.blocks import bundle, unbundle
from spinn.util.blocks import treelstm, expand_along, dropout
from spinn.util.blocks import var_mean
from spinn.util.blocks import BaseSentencePairTrainer, HeKaimingInit

from sklearn import metrics

import spinn.cbow


T_SHIFT  = 0
T_REDUCE = 1
T_SKIP   = 2


class SentencePairTrainer(BaseSentencePairTrainer):
    def init_params(self, **kwargs):
        for name, param in self.model.namedparams():
            data = param.data
            print("Init: {}:{}".format(name, data.shape))
            if len(data.shape) >= 2:
                data[:] = HeKaimingInit(data.shape)
            else:
                data[:] = np.random.uniform(-0.1, 0.1, data.shape)

    def init_optimizer(self, lr=0.01, l2_lambda=0.0, **kwargs):
        relevant_params = [w for w in self.model.parameters() if w.requires_grad]
        self.optimizer = optim.Adam(relevant_params, lr=lr, betas=(0.9, 0.999), weight_decay=l2_lambda)


class SentenceTrainer(SentencePairTrainer):
    pass


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, predict, tracker_dropout_rate=0.0, use_skips=False):
        super(Tracker, self).__init__()
        self.lateral = nn.Linear(tracker_size, 4 * tracker_size)
        self.buf = nn.Linear(size, 4 * tracker_size, bias=False)
        self.stack1 = nn.Linear(size, 4 * tracker_size, bias=False)
        self.stack2 = nn.Linear(size, 4 * tracker_size, bias=False)
        if predict:
            self.transition = nn.Linear(tracker_size, 3 if use_skips else 2)
        self.state_size = tracker_size
        self.tracker_dropout_rate = tracker_dropout_rate
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, bufs, stacks, train):
        self.batch_size = len(bufs)
        zeros = Variable(np.zeros(bufs[0][0].shape, dtype=bufs[0][0].data.dtype),
                         volatile=not train)
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
                volatile=not train)

        lstm_in = dropout(lstm_in, self.tracker_dropout_rate, train=train)

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


    def __call__(self, example, train, print_transitions=False, use_internal_parser=False,
                 validate_transitions=True, use_random=False,
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
                        use_random=use_random,
                        use_reinforce=use_reinforce,
                        rl_style=rl_style,
                        rl_baseline=rl_baseline,
                        )

    def run(self, train, print_transitions=False, run_internal_parser=False, use_internal_parser=False,
            validate_transitions=True, use_random=False, use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
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
                raise Exception('Running without transitions not implemented')

            cant_skip = np.array([t != T_SKIP for t in transitions])
            if hasattr(self, 'tracker') and (self.use_skips or sum(cant_skip) > 0):
                transition_hyp = self.tracker(self.bufs, self.stacks, train)
                if transition_hyp is not None and run_internal_parser:
                    transition_hyp = to_cpu(transition_hyp)
                    if hasattr(self, 'transitions'):
                        memory = {}
                        truth_acc = transitions
                        hyp_xent = transition_hyp
                        if use_reinforce:
                            probas = F.softmax(transition_hyp)
                            samples = np.array([T_SKIP for _ in self.bufs], dtype=np.int32)
                            samples[cant_skip] = [np.random.choice(self.choices, 1, p=proba)[0] for proba in probas.data[cant_skip]]

                            transition_preds = samples
                            hyp_acc = probas
                            truth_xent = samples
                        else:
                            transition_preds = transition_hyp.data.argmax(axis=1)
                            hyp_acc = transition_hyp
                            truth_xent = transitions

                        if use_random:
                            print("Using random")
                            transition_preds = np.random.choice(self.choices, len(self.bufs))
                        
                        if validate_transitions:
                            transition_preds = self.validate(transition_arr, transition_preds,
                                self.stacks, self.buffers_t, self.buffers_n)

                        memory["logits"] = transition_hyp
                        memory["preds"]  = transition_preds

                        if not self.use_skips:
                            hyp_acc = hyp_acc.data[cant_skip]
                            truth_acc = truth_acc[cant_skip]

                            cant_skip_mask = np.tile(np.expand_dims(cant_skip, axis=1), (1, 2))
                            hyp_xent = F.split_axis(transition_hyp, transition_hyp.shape[0], axis=0)
                            hyp_xent = F.concat([hyp_xent[iii] for iii, y in enumerate(cant_skip) if y], axis=0)
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
                            zeros = Variable(np.zeros(buf[0].shape,
                                dtype=buf[0].data.dtype),
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
                        assert isinstance(new_stack_item.data, np.ndarray), "Pushing cupy array to stack"
                        stack.append(new_stack_item)

        if self.transition_weight is not None:
            # We compute statistics after the fact, since sub-batches can
            # have different sizes when not using skips.
            hyp_acc, truth_acc, hyp_xent, truth_xent = self.get_statistics()

            transition_acc = F.accuracy(
                hyp_acc, truth_acc.astype(np.int32))

            transition_loss = F.softmax_cross_entropy(
                hyp_xent, truth_acc.astype(np.int32),
                normalize=False)

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
            F.squeeze(F.concat([F.expand_dims(ss, 1) for ss in s], axis=0))
            if isinstance(s[0], Variable) else
            np.array(reduce(lambda x, y: x + y.tolist(), s, []))
            for s in statistics]

        hyp_acc, truth_acc, hyp_xent, truth_xent = statistics
        return hyp_acc, truth_acc, hyp_xent, truth_xent


class BaseModel(nn.Module):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
                 input_keep_rate, classifier_keep_rate,
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
        
        # Initialize Classifier Parameters
        self.init_mlp(mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn)
        self.mlp_input_dim = mlp_input_dim
        self.mlp_dim = mlp_dim
        self.num_mlp_layers = num_mlp_layers
        self.mlp_bn = mlp_bn

        # RL Params
        self.reinforce_lr = 0.01
        self.mu = 0.1
        self.baseline = 0

        if rl_baseline == "policy":
            baseline_model_module = spinn.cbow

            if use_sentence_pair:
                baseline_model_cls = baseline_model_module.SentencePairModel
            else:
                baseline_model_cls = baseline_model_module.SentenceModel

            _model_dim = word_embedding_dim
            _num_classes = 1 # Reward will be between 0 and 1

            # TODO:
            # The policy net is naively initialized. Certain features
            # such as keep_rate, batch_norm, num_mlp_layers,
            # etc. are simply taken from the hyperparams. We might
            # want these to be different.
            self.add_link("policy", baseline_model_cls(_model_dim, word_embedding_dim, vocab_size,
             seq_length, initial_embeddings, _num_classes,
             mlp_dim=rl_policy_dim,
             input_keep_rate=input_keep_rate,
             classifier_keep_rate=classifier_keep_rate,
             use_input_norm=use_input_norm,
             tracker_keep_rate=tracker_keep_rate,
             tracking_lstm_hidden_dim=tracking_lstm_hidden_dim,
             transition_weight=transition_weight,
             use_tracking_lstm=use_tracking_lstm,
             use_tracking_in_composition=use_tracking_in_composition,
             use_sentence_pair=use_sentence_pair,
             num_mlp_layers=num_mlp_layers,
             mlp_bn=mlp_bn,
             gpu=gpu,
             use_skips=use_skips,
             use_encode=False,
             projection_dim=projection_dim,
             use_difference_feature=use_difference_feature,
             use_product_feature=use_product_feature,
            ))

        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        self.initial_embeddings = initial_embeddings
        self.classifier_dropout_rate = 1. - classifier_keep_rate
        self.word_embedding_dim = word_embedding_dim
        self.use_encode = use_encode
        self.transition_weight = transition_weight

        if projection_dim <= 0 or not self.use_encode:
            projection_dim = self.stack_hidden_dim

        args = {
            'size': projection_dim,
            'tracker_size': tracking_lstm_hidden_dim if use_tracking_lstm else None,
            'use_tracking_in_composition': use_tracking_in_composition,
            'transition_weight': transition_weight,
            'input_dropout_rate': 1. - input_keep_rate,
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

        self.spinn = SPINN(args, vocab, use_skips=use_skips)

        # TODO: Add encoding layer.
        if self.use_encode:
            raise NotImplementedError()

        self.init_params()
        print(self)


    def init_params(self):
        initrange = 0.1
        for w in self.parameters():
            if w.requires_grad:
                print(w.size())
                if len(w.size()) >= 2:
                    w.data.set_(torch.from_numpy(HeKaimingInit(w.data.size())).float())
                else:
                    w.data.uniform_(-initrange, initrange)


    def init_mlp(self, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers, mlp_bn):
        features_dim = mlp_input_dim
        if self.use_difference_feature:
            features_dim += self.stack_hidden_dim
        if self.use_product_feature:
            features_dim += self.stack_hidden_dim
        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), nn.Linear(features_dim, mlp_dim))
            if mlp_bn:
                # TODO: Add batch norm in semantic classifier.
                raise NotImplementedError()
            features_dim = mlp_dim
        setattr(self, 'l{}'.format(num_mlp_layers), nn.Linear(features_dim, num_classes))


    def build_example(self, sentences, transitions, train):
        raise Exception('Not implemented.')


    def run_embed(self, example, train):
        embeds = self.embed(example.tokens, train)

        b, l = example.tokens.shape[:2]

        embeds = F.split_axis(to_cpu(embeds), b, axis=0, force_tuple=True)
        embeds = [F.expand_dims(x, 0) for x in embeds]
        embeds = F.concat(embeds, axis=0)

        if self.use_encode:
            _, _, fwd_hs = self.fwd_rnn(embeds, train, keep_hs=True)
            _, _, bwd_hs = self.bwd_rnn(embeds, train, keep_hs=True, reverse=True)
            hs = F.concat([fwd_hs, bwd_hs], axis=2)
            embeds = hs

        embeds = [F.split_axis(x, l, axis=0, force_tuple=True) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        assert b == len(buffers)

        example.tokens = buffers

        return example


    def run_spinn(self, example, train, use_internal_parser,
                  validate_transitions=True, use_random=False,
                  use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h_both, transition_loss, transition_acc = self.spinn(example, train,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions,
                               use_random=use_random,
                               use_reinforce=use_reinforce,
                               rl_style=rl_style,
                               rl_baseline=rl_baseline,
                               )
        return h_both, transition_acc, transition_loss


    def run_mlp(self, h, train):
        # Pass through MLP Classifier.
        batch_size = h.shape[:2]

        if self.use_sentence_pair:
            prem, hyp = h[:, :self.stack_hidden_dim], h[:, self.stack_hidden_dim*2:self.stack_hidden_dim*3]
            h = F.concat([prem, hyp], axis=1)  # Strip off 'c' states.   
            if self.use_difference_feature:
                h = F.concat([h, prem - hyp], axis=1)
            
            if self.use_product_feature:
                h = F.concat([h, prem * hyp], axis=1)

        else:
            h = h[:, :self.stack_hidden_dim]  # Strip off 'c' states.   

        h = to_gpu(h)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if self.mlp_bn:
                bn = getattr(self, 'bn{}'.format(i))
                h = bn(h, test=not train, finetune=False)
            # TODO: Theano code rescales during Eval. This is opposite of what Chainer does.
            h = dropout(h, ratio=self.classifier_dropout_rate, train=train)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        y = layer(h)
        return y


    def __call__(self, sentences, transitions, y_batch=None, train=True,
                 use_reinforce=False, rl_style="zero-one", rl_baseline="ema",
                 use_internal_parser=False, validate_transitions=True, use_random=False):
        example = self.build_example(sentences, transitions, train)
        example_embed = self.run_embed(example, train)
        h, transition_acc, transition_loss = self.run_spinn(example_embed, train, use_internal_parser,
            validate_transitions, use_random, use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        if y_batch is not None:
            accum_loss = self.classifier(y, Variable(y_batch, volatile=not train), train)
            self.accuracy = self.accFun(y, self.__mod.array(y_batch))
            acc = self.accuracy.data
        else:
            accum_loss = 0.0
            acc = 0.0

        if train and use_reinforce:
            rewards = self.build_rewards(y, y_batch, rl_style)

            if rl_baseline == "ema": # Exponential Moving Average
                self.baseline = self.baseline * (1 - self.mu) + self.mu * np.mean(rewards)
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
                raise NotImplementedError("Not implemented.")

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

        return y, accum_loss, acc, transition_acc, transition_loss, rl_loss


    def run_policy(self, sentences, transitions, y_batch, train, rewards, rl_style):
        if rl_style != "zero-one":
            raise NotImplementedError("Policy net is only compatible with zero-one loss right now."
                "It predicts a single value between 0 and 1.")
        y, _, _, _, _ = self.policy(sentences, transitions, y_batch=None, train=train)
        pred_reward = F.flatten(F.sigmoid(y)) # Squash between 0 and 1
        return pred_reward, F.mean_squared_error(pred_reward, rewards)


    def run_greedy_max(self, sentences, transitions, y_batch, train, rewards, rl_style):
        # TODO: Should this be run with train=False? Will effect batchnorm and dropout.
        y, _, _, _, _ = self.__call__(sentences, transitions, y_batch=None, train=train, use_internal_parser=True)
        pred_reward = self.build_rewards(y, y_batch, rl_style)
        return pred_reward


    def build_rewards(self, logits, y, style="zero-one"):
        if style == "xent":
            rewards = -1. * F.concat([F.expand_dims(
                        F.softmax_cross_entropy(logits[i:(i+1)], y[i:(i+1)]), axis=0)
                        for i in range(y.shape[0])], axis=0).data
        elif style == "zero-one":
            rewards = (F.argmax(logits, axis=1).data == y).astype(np.float32)
        else:
            raise Exception("Not implemented")
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
        log_p_preds = F.select_item(log_p, truth_xent)

        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            rewards = np.concatenate([rewards, rewards], axis=0)
        else:
            assert self.spinn.transition_mask.shape[0] == rewards.shape[0]

        # Expand rewards
        if self.spinn.use_skips:
            rewards = expand_along(rewards, np.full(self.spinn.transition_mask.shape, True))
        else:
            rewards = expand_along(rewards, self.spinn.transition_mask)

        rl_loss = F.sum(-1. * log_p_preds * rewards) / log_p_preds.shape[0]

        return rl_loss


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


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True,
                  use_random=False, use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h_both, transition_acc, transition_loss = super(SentencePairModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions,
            use_random, use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
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


    def run_spinn(self, example, train, use_internal_parser=False, validate_transitions=True,
                  use_random=False, use_reinforce=False, rl_style="zero-one", rl_baseline="ema"):
        h, transition_acc, transition_loss = super(SentenceModel, self).run_spinn(
            example, train, use_internal_parser, validate_transitions,
            use_random, use_reinforce=use_reinforce, rl_style=rl_style, rl_baseline=rl_baseline)
        h = F.concat(h, axis=0)
        return h, transition_acc, transition_loss
