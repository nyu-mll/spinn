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


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, lateral_tracking=True):
        super(Tracker, self).__init__()

        # Initialize layers.
        self.buf = Linear()(size, 4 * tracker_size, bias=True)
        self.stack1 = Linear()(size, 4 * tracker_size, bias=False)
        self.stack2 = Linear()(size, 4 * tracker_size, bias=False)

        if lateral_tracking:
            self.lateral = Linear(initializer=HeKaimingInitializer)(tracker_size, 4 * tracker_size, bias=False)
        else:
            self.transform = Linear(initializer=HeKaimingInitializer)(4 * tracker_size, tracker_size)

        self.lateral_tracking = lateral_tracking
        self.state_size = tracker_size

        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, top_buf, top_stack_1, top_stack_2):
        tracker_inp = self.buf(top_buf)
        tracker_inp += self.stack1(top_stack_1)
        tracker_inp += self.stack2(top_stack_2)

        batch_size = tracker_inp.size(0)

        if self.lateral_tracking:
            if self.h is not None:
                tracker_inp += self.lateral(self.h)
            if self.c is None:
                self.c = to_gpu(Variable(torch.from_numpy(
                    np.zeros((batch_size, self.state_size),
                                  dtype=np.float32)),
                    volatile=tracker_inp.volatile))

            # Run tracking lstm.
            self.c, self.h = lstm(self.c, tracker_inp)

            return self.h, self.c
        else:
            outp = self.transform(tracker_inp.clamp(min=0)).clamp(min=0)
            return outp, None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class SPINN(nn.Module):
    """Applies a TreeLSTM Recursive Neural Network to an input sequence.

    Composition uses the following function:

    .. math::

        \begin{array}{ll}
        x_t = W_{hlx} hl_{t-1} + W_{hrx} hr_{t-1}] \\
        i_t, fl_t, fr_t, o_t = sigmoid(slice(x_t, 0, 4)) \\
        z_t = \tanh(slice(x_t, 4, 5)) \\
        c_t = fl_t * cl_{t-1} + fr_t * cr_{t-1} + i_t * z_t \\
        h_t = o_t * \tanh(c_t)
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell state at time `t`.

    Args:
        size: The number of features in the hidden state h

    Inputs: x, t
        - **x** (batch, seq_len, size): tensor containing the features of the input sequence.
        - **t** (batch, seq_len * 2 - 1): tensor containing the transitions used to parse the input sequence.

    Outputs: output, (h_n, c_n)
        - **output** (batch, seq_len * 2 - 1, size * 3): tensor containing
          the input features `(buf_1_t, stack_1_t, stack_2_t)` from SPINN, for each t.
        - **h_n** (batch, size): tensor containing the hidden state for t=seq_len
        - **c_n** (batch, size): tensor containing the cell state for t=seq_len

    Attributes:
        weight_hl_x : the learnable input-hidden weights of shape `(size x 5*size)`
        weight_hr_x : the learnable input-hidden weights of shape `(size x 5*size)`
        bias_hl_x   : the learnable input-hidden bias of shape `(5*size)`
        bias_hr_x   : the learnable input-hidden bias of shape `(5*size)`

    Examples::

        >>> spinn = SPINN(10)
        >>> x = Variable(torch.randn(2, 3, 10))
        >>> t = Variable(torch.LongTensor([
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
        ]))
        >>> output, (h_n, c_n) = spinn(x, t)
    """

    def __init__(self, args, vocab, predict_use_cell):
        super(SPINN, self).__init__()

        # Optional debug mode.
        self.debug = False

        self.transition_weight = args.transition_weight

        self.wrap_items = args.wrap_items
        self.extract_h = args.extract_h

        # Reduce function for semantic composition.
        self.reduce = args.composition
        if args.tracker_size is not None:
            self.tracker = Tracker(args.size, args.tracker_size, args.lateral_tracking)
            if args.transition_weight is not None:
                # TODO: Might be interesting to try a different network here.
                self.predict_use_cell = predict_use_cell
                tinp_size = args.tracker_size * 2 if predict_use_cell else args.tracker_size
                self.transition_net = nn.Linear(tinp_size, 2)

        self.choices = np.array([T_SHIFT, T_REDUCE], dtype=np.int32)

        self.shift_probabilities = ShiftProbabilities()

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.n_tokens = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.n_tokens), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = self.zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Initialize Buffers. Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:] for b, b_n in zip(self.bufs, self.n_tokens)]

        # Initialize Stacks.
        self.stacks = [[zeros, zeros] for buf in self.bufs]

        # Initialize other.
        self.n_reduces = np.zeros(len(self.bufs), dtype=np.int32)
        self.n_steps = np.zeros(len(self.bufs), dtype=np.int32)

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, bufs, zero_padded=True):
        # Note: There is one zero added to bufs, and two zeros added to stacks.
        # Make sure to adjust for this if using lengths of either.
        buf_adjust = 1 if zero_padded else 0
        stack_adjust = 2 if zero_padded else 0

        _transitions = np.array(transitions)
        _preds = preds.copy()
        _invalid = np.zeros(preds.shape, dtype=np.bool)

        incorrect = 0
        cant_skip = _transitions != T_SKIP
        must_skip = _transitions == T_SKIP

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError("Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        check_mask = np.logical_and(cant_skip, must_shift)
        _invalid += np.logical_and(_preds != T_SHIFT, check_mask)
        _preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        check_mask = np.logical_and(cant_skip, must_reduce)
        _invalid += np.logical_and(_preds != T_REDUCE, check_mask)
        _preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        _preds[must_skip] = T_SKIP

        return _preds, _invalid

    def predict_actions(self, transition_output):
        transition_dist = F.log_softmax(transition_output)
        transition_dist = transition_dist.data.cpu().numpy()
        transition_preds = transition_dist.argmax(axis=1)
        return transition_preds

    def get_transitions_per_example(self, style="preds"):
        if style == "preds":
            source = "t_preds"
        elif style == "given":
            source = "t_given"
        else:
            raise NotImplementedError

        t_preds = np.concatenate([m['t_preds'] for m in self.memories if m.get('t_preds', None) is not None])
        t_preds = torch.from_numpy(t_preds).long()
        t_logits = torch.cat([m['t_logits'] for m in self.memories if m.get('t_logits', None) is not None], 0).data.cpu()
        t_logits = torch.cat([t_logits, torch.zeros(t_logits.size(0), 1)], 1)
        t_strength = torch.gather(t_logits, 1, t_preds.view(-1, 1))

        _transitions = [m[source].reshape(1, -1) for m in self.memories if m.get(source, None) is not None]
        transitions = np.concatenate(_transitions).T

        t_strength = torch.exp(t_strength.view(*list(reversed(transitions.shape))).t())

        skip_mask = (torch.from_numpy(transitions) == T_SKIP).byte()
        t_strength[skip_mask] = 0.

        return transitions, t_strength

    def t_shift(self, buf, stack, tracking, buf_tops, trackings):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop() if len(buf) > 0 else self.zeros)
        trackings.append(tracking)

    def t_reduce(self, buf, stack, tracking, lefts, rights, trackings):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            if len(stack) > 0:
                reduce_inp.append(stack.pop())
            else:
                if self.debug:
                    raise IndexError
                # If we try to Reduce, but there are less than 2 items on the stack,
                # then treat any available item as the right input, and use zeros
                # for any other inputs.
                # NOTE: Only happens on cropped data.
                reduce_inp.append(self.zeros)

        trackings.append(tracking)

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""
        pass

    def shift_phase(self, tops, trackings, stacks, idxs):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def reduce_phase(self, lefts, rights, trackings, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(
                lefts, rights, trackings))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        transition_loss = None
        transition_acc = 0.0
        num_transitions = inp_transitions.shape[1]
        batch_size = inp_transitions.shape[0]
        invalid_count = np.zeros(batch_size)

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)
            sub_batch_size = len(transition_arr)

            # A mask based on SKIP transitions.
            cant_skip = np.array(transitions) != T_SKIP
            must_skip = np.array(transitions) == T_SKIP

            # Memories
            # ========
            # Keep track of key values to determine accuracy and loss.
            self.memory = {}

            # Prepare tracker input.
            if self.debug and any(len(buf) < 1 or len(stack) for buf, stack in zip(self.bufs, self.stacks)):
                # To elaborate on this exception, when cropping examples it is possible
                # that your first 1 or 2 actions is a reduce action. It is unclear if this
                # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                # turn on the truncate batch flag, and set the eval_seq_length very high.
                raise IndexError("Warning: You are probably trying to encode examples"
                      "with cropped transitions. Although, this is a reasonable"
                      "feature, when predicting/validating transitions, you"
                      "probably will not get the behavior that you expect. Disable"
                      "this exception if you dare.")
            self.memory['top_buf'] = self.wrap_items([buf[-1] if len(buf) > 0 else self.zeros for buf in self.bufs])
            self.memory['top_stack_1'] = self.wrap_items([stack[-1] if len(stack) > 0 else self.zeros for stack in self.stacks])
            self.memory['top_stack_2'] = self.wrap_items([stack[-2] if len(stack) > 1 else self.zeros for stack in self.stacks])

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if hasattr(self, 'tracker') and sum(cant_skip) > 0:

                # Get hidden output from the tracker. Used to predict transitions.
                tracker_h, tracker_c = self.tracker(
                    self.extract_h(self.memory['top_buf']),
                    self.extract_h(self.memory['top_stack_1']),
                    self.extract_h(self.memory['top_stack_2']))

                if hasattr(self, 'transition_net'):
                    transition_inp = [tracker_h]
                    if self.predict_use_cell:
                        transition_inp += [tracker_c]
                    transition_inp = torch.cat(transition_inp, 1)
                    transition_output = self.transition_net(transition_inp)

                if hasattr(self, 'transition_net') and run_internal_parser:

                    # Predict Actions
                    # ===============

                    # Distribution of transitions use to calculate transition loss.
                    self.memory["t_logits"] = F.log_softmax(transition_output)

                    # Given transitions.
                    self.memory["t_given"] = transitions

                    # TODO: Mask before predicting. This should simplify things and reduce computation.
                    # The downside is that in the Action Phase, need to be smarter about which stacks/bufs
                    # are selected.
                    transition_preds = self.predict_actions(transition_output)

                    # Constrain to valid actions
                    # ==========================

                    validated_preds, invalid_mask = self.validate(transition_arr, transition_preds, self.stacks, self.bufs)
                    if validate_transitions:
                        transition_preds = validated_preds

                    # Keep track of which predictions have been valid.
                    self.memory["t_valid_mask"] = np.logical_not(invalid_mask)
                    invalid_count += invalid_mask

                    # If the given action is skip, then must skip.
                    transition_preds[must_skip] = T_SKIP

                    # Actual transition predictions. Used to measure transition accuracy.
                    self.memory["t_preds"] = transition_preds

                    # Binary mask of examples that have a transition.
                    self.memory["t_mask"] = cant_skip

                    # If this FLAG is set, then use the predicted actions rather than the given.
                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

            # Pre-Action Phase
            # ================

            # For SHIFT
            s_stacks, s_tops, s_trackings, s_idxs = [], [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights, r_trackings = [], [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None))

            for batch_idx, (transition, buf, stack, tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    self.t_shift(buf, stack, tracking, s_tops, s_trackings)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE: # reduce
                    self.t_reduce(buf, stack, tracking, r_lefts, r_rights, r_trackings)
                    r_stacks.append(stack)
                elif transition == T_SKIP: # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_trackings, s_stacks, s_idxs)
            self.reduce_phase(r_lefts, r_rights, r_trackings, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_trackings, r_stacks)

            # Memory Phase
            # ============

            # APPEND ALL MEMORIES. MASK LATER.

            self.memories.append(self.memory)

            # Update number of reduces seen so far.
            self.n_reduces += (np.array(transition_arr) == T_REDUCE)

            # Update number of non-skip actions seen so far.
            self.n_steps += (np.array(transition_arr) != T_SKIP)

        # Loss Phase
        # ==========

        if hasattr(self, 'tracker') and hasattr(self, 'transition_net'):
            t_preds = np.concatenate([m['t_preds'] for m in self.memories if m.get('t_preds', None) is not None])
            t_given = np.concatenate([m['t_given'] for m in self.memories if m.get('t_given', None) is not None])
            t_mask = np.concatenate([m['t_mask'] for m in self.memories if m.get('t_mask', None) is not None])
            t_logits = torch.cat([m['t_logits'] for m in self.memories if m.get('t_logits', None) is not None], 0)

            # We compute accuracy and loss after all transitions have complete,
            # since examples can have different lengths when not using skips.

            # Transition Accuracy.
            n = t_mask.shape[0]
            n_skips = n - t_mask.sum()
            n_total = n - n_skips
            n_correct = (t_preds == t_given).sum() - n_skips
            transition_acc = n_correct / float(n_total)

            # Transition Loss.
            index = to_gpu(Variable(torch.from_numpy(np.arange(t_mask.shape[0])[t_mask])).long())
            select_t_given = to_gpu(Variable(torch.from_numpy(t_given[t_mask]), volatile=not self.training).long())
            select_t_logits = torch.index_select(t_logits, 0, index)
            transition_loss = nn.NLLLoss()(select_t_logits, select_t_given) * self.transition_weight

            self.n_invalid = (invalid_count > 0).sum()
            self.invalid = self.n_invalid / float(batch_size)

        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."

        return [stack[-1] for stack in self.stacks], transition_acc, transition_loss
