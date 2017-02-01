import unittest
import argparse

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN


def default_args():
    model_dim = 10
    use_tracking_lstm = True
    transition_weight = 1.0
    use_history = False
    save_stack = False
    input_keep_rate = 1.0
    use_input_dropout = False
    use_input_norm = False
    use_tracker_dropout = False
    tracker_dropout_rate = 0.0
    tracking_lstm_hidden_dim = 4

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

    return args


def default_embeddings(vocab_size, embedding_dim):
    initial_embeddings = np.arange(vocab_size).reshape(
            (vocab_size, 1)).repeat(embedding_dim, axis=1).astype(np.float32)

    vocab = {
        'size': initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size,
        'vectors': initial_embeddings,
    }
    vocab = argparse.Namespace(**vocab)

    return vocab, initial_embeddings


def build_example(tokens, transitions, train=False):
    example = {
        'tokens': Variable(tokens, volatile=not train),
        'transitions': transitions
    }
    example = argparse.Namespace(**example)
    return example


class SPINNTestCase(unittest.TestCase):

    def _setup(self, batch_size=2, seq_length=4):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim = 3
        self.vocab_size = vocab_size = 10
        self.seq_length = seq_length
        args = default_args()
        vocab, initial_embeddings = default_embeddings(vocab_size, embedding_dim)
        self.model = SPINN(args, vocab, use_reinforce=False)

    def xtest_basic_stack(self):
        self._setup(seq_length=4)

        train = False

        X = np.array([
            [3, 1,  2, 0],
            [3, 2,  4, 5]
        ], dtype=np.int32)

        transitions = np.array([
            # First input: push a bunch onto the stack
            [0, 0, 0, 0],
            # Second input: push, then merge, then push more. (Leaves one item
            # on the buffer.)
            [0, 0, 1, 0]
        ], dtype=np.int32)

        expected_stack_lens = [4, 2]

        example = build_example(X, transitions)
        _ = self.model(example)
        stack_lens = [len(s) for s in self.model.stacks]\

        np.testing.assert_equal(stack_lens, expected_stack_lens)


if __name__ == '__main__':
    unittest.main()
