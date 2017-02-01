import unittest

import numpy as np

import os
from spinn import util
from spinn.data.snli import load_snli_data
from spinn.data.sst import load_sst_data
from spinn.data.arithmetic import load_simple_data
from spinn.data.boolean import load_boolean_data
from collections import Counter


""" FAQ

1. How are sentences padded?

Sentences are padded to seq_length. Sentences with associated transitions that are
longer than seq_length are thrown away. Sentences are padded with a special
sentence padding token. Sentences are padded on the right.

2. How many transitions are there?

There are 2N - 1 transitions where N is the length of the associated sentence.

3. How are transitions padded?

Transitions are padded to seq_length using a transition padding symbol (different
from the sentence padding token). Transitions are padded on the left.

"""


snli_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_snli.jsonl")
sst_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_sst.txt")
arithmetic_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_simple.tsv")
boolean_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_boolean.tsv")
embedding_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_embedding_matrix.5d.txt")
word_embedding_dim = 5


class MockLogger(object):
    def Log(self, *args, **kwargs):
        pass


def t_is_left_padded(ts):
    assert len([t for t in ts if t == util.SKIP_SYMBOL]) > 0, \
        "Transitions must be padded for this check to work"

    assert ts[0] != ts[-1], "This check does not work for transitions padded on both ends."

    return ts[0] == util.SKIP_SYMBOL


def t_is_left_to_right(ts):
    for t in ts:
        if t == util.SKIP_SYMBOL:
            continue
        if t == util.SHIFT_SYMBOL:
            return True
        else:
            # If the first symbol is a REDUCE, then the transitions are reversed.
            return False

    # Hypothetically, is possible to have zero transitions, and padded completely with SKIPs.
    # That being said, this should never happen, so return False.
    return False


def s_is_left_padded(s):
    assert len([w for w in s if w == util.SENTENCE_PADDING_SYMBOL]) > 0, \
        "Sentence must be padded for this check to work"

    assert s[0] != s[-1], "This check does not work for sentences padded on both ends."

    return s[0] == util.SENTENCE_PADDING_SYMBOL


def s_is_left_to_right(s, EOS_TOKEN):
    if not isinstance(EOS_TOKEN, list):
        EOS_TOKEN = [EOS_TOKEN]

    for w in s:
        if w == util.SENTENCE_PADDING_SYMBOL:
            continue

        # TODO: What about single token sentence? This is probably good enough for now.
        if w in EOS_TOKEN:
            # If the first symbol is an EOS, then the sentence is reversed.
            return False
        else:
            return True

    # Hypothetically, is possible to have all padding.
    # That being said, this should never happen, so return False.
    return False

        
class DataTestCase(unittest.TestCase):

    def test_vocab(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(snli_data_path)
        data_sets = [(snli_data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        assert len(vocabulary) == 10

    def test_load_embed(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(snli_data_path)
        data_sets = [(snli_data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, word_embedding_dim, embedding_data_path)
        assert initial_embeddings.shape == (10, 5)


class SNLITestCase(unittest.TestCase):

    def test_load(self):
        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(snli_data_path)
        assert len(raw_data) == 20

        hyp_seq_lengths = Counter([len(x['hypothesis_transitions'])
                        for x in raw_data])
        assert hyp_seq_lengths == {13: 4, 15: 4, 11: 2, 17: 2, 23: 2, 5: 1, 39: 1, 9: 1, 19: 1, 7: 1, 29: 1}

        prem_seq_lengths = Counter([len(x['premise_transitions'])
                        for x in raw_data])
        assert prem_seq_lengths == {35: 8, 67: 3, 19: 3, 53: 3, 33: 3}

        min_seq_lengths = Counter([min(len(x['hypothesis_transitions']), len(x['premise_transitions']))
                        for x in raw_data])
        assert min_seq_lengths == {5: 1, 7: 1, 9: 1, 11: 2, 13: 4, 15: 4, 17: 2, 19: 2, 23: 2, 35: 1}

    def test_preprocess(self):
        seq_length = 25
        for_rnn = False
        use_left_padding = True

        data_manager = load_snli_data
        raw_data, _ = data_manager.load_data(snli_data_path)
        data_sets = [(snli_data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, word_embedding_dim, embedding_data_path)

        EOS_TOKEN = vocabulary["."]

        data = util.PreprocessDataset(
            raw_data, vocabulary, seq_length, data_manager, eval_mode=False, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=for_rnn, use_left_padding=use_left_padding)

        tokens, transitions, labels, num_transitions = data
        
        # Filter examples that don't have lengths <= seq_length
        assert tokens.shape == (2, seq_length, 2)
        assert transitions.shape == (2, seq_length, 2)

        for s, ts, (num_hyp_t, num_prem_t) in zip(tokens, transitions, num_transitions):
            hyp_s = s[:, 0]
            prem_s = s[:, 1]
            hyp_t = ts[:, 0]
            prem_t = ts[:, 1]

            # The sentences should start with a word and end with an EOS.
            assert s_is_left_to_right(hyp_s, EOS_TOKEN)
            assert s_is_left_to_right(prem_s, EOS_TOKEN)

            # The sentences should be padded on the right.
            assert not s_is_left_padded(hyp_s)
            assert not s_is_left_padded(prem_s)
            
            # The num_transitions should count non-skip transitions
            assert len([x for x in hyp_t if x != util.SKIP_SYMBOL]) == num_hyp_t
            assert len([x for x in prem_t if x != util.SKIP_SYMBOL]) == num_prem_t

            # The transitions should start with SKIP and end with REDUCE (ignoring SKIPs).
            assert t_is_left_to_right(hyp_t)
            assert t_is_left_to_right(prem_t)

            # The transitions should be padded on the left.
            assert t_is_left_padded(hyp_t)
            assert t_is_left_padded(prem_t)


class SSTTestCase(unittest.TestCase):

    def test_load(self):
        data_manager = load_sst_data
        raw_data, _ = data_manager.load_data(sst_data_path)
        assert len(raw_data) == 20

        seq_lengths = Counter([len(x['transitions']) for x in raw_data])
        assert seq_lengths == {57: 3, 37: 2, 47: 2, 45: 2, 15: 2, 53: 2, 59: 2, 65: 1, 67: 1, 23: 1, 35: 1, 25: 1}

    def test_preprocess(self):
        seq_length = 30
        for_rnn = False
        use_left_padding = True

        data_manager = load_sst_data
        raw_data, _ = data_manager.load_data(sst_data_path)
        data_sets = [(sst_data_path, raw_data)]
        vocabulary = util.BuildVocabulary(
            raw_data, data_sets, embedding_data_path, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, word_embedding_dim, embedding_data_path)

        EOS_TOKEN = vocabulary["."]

        data = util.PreprocessDataset(
            raw_data, vocabulary, seq_length, data_manager, eval_mode=False, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=for_rnn, use_left_padding=use_left_padding)

        tokens, transitions, labels, num_transitions = data

        # Filter examples that don't have lengths <= seq_length
        assert tokens.shape == (4, seq_length)
        assert transitions.shape == (4, seq_length)

        for s, ts, num_t in zip(tokens, transitions, num_transitions):

            # The sentences should start with a word and end with an EOS.
            assert s_is_left_to_right(s, EOS_TOKEN)

            # The sentences should be padded on the right.
            assert not s_is_left_padded(s)
            
            # The num_transitions should count non-skip transitions
            assert len([x for x in ts if x != util.SKIP_SYMBOL]) == num_t

            # The transitions should start with SKIP and end with REDUCE (ignoring SKIPs).
            assert t_is_left_to_right(ts)

            # The transitions should be padded on the left.
            assert t_is_left_padded(ts)


class ArithmeticTestCase(unittest.TestCase):

    def test_load(self):
        # NOTE: Arithmetic tsv file uses a tab between label and example.
        data_manager = load_simple_data
        raw_data, _ = data_manager.load_data(arithmetic_data_path)
        assert len(raw_data) == 20

        seq_lengths = Counter([len(x['transitions']) for x in raw_data])
        assert seq_lengths == {5: 12, 9: 4, 13: 3, 17: 1}

    def test_preprocess(self):
        seq_length = 10
        for_rnn = False
        use_left_padding = True

        data_manager = load_simple_data
        raw_data, vocabulary = data_manager.load_data(arithmetic_data_path)

        OPERATOR_TOKENS = [vocabulary["+"], vocabulary["-"]]

        data = util.PreprocessDataset(
            raw_data, vocabulary, seq_length, data_manager, eval_mode=False, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=for_rnn, use_left_padding=use_left_padding)

        tokens, transitions, labels, num_transitions = data

        # Filter examples that don't have lengths <= seq_length
        assert tokens.shape == (16, seq_length)
        assert transitions.shape == (16, seq_length)

        for s, ts, num_t in zip(tokens, transitions, num_transitions):

            # The sentence should begin with an operator.
            assert any(s[0] == tkn for tkn in OPERATOR_TOKENS)

            # The sentences should be padded on the right.
            assert not s_is_left_padded(s)

            # The num_transitions should count non-skip transitions
            assert len([x for x in ts if x != util.SKIP_SYMBOL]) == num_t

            # The transitions should start with SKIP and end with REDUCE (ignoring SKIPs).
            assert t_is_left_to_right(ts)

            # The transitions should be padded on the left.
            assert t_is_left_padded(ts)


class BooleanTestCase(unittest.TestCase):

    def test_load(self):
        # NOTE: Boolean tsv file uses a tab between label and example.
        data_manager = load_boolean_data
        raw_data, _ = data_manager.load_data(boolean_data_path)
        assert len(raw_data) == 20

        seq_lengths = Counter([len(x['transitions']) for x in raw_data])
        assert seq_lengths == {29: 7, 21: 5, 27: 3, 25: 3, 23: 2}

    def test_preprocess(self):
        seq_length = 24
        for_rnn = False
        use_left_padding = True

        data_manager = load_boolean_data
        raw_data, vocabulary = data_manager.load_data(boolean_data_path)

        OPERATOR_TOKENS = [vocabulary["and"], vocabulary["or"]]

        data = util.PreprocessDataset(
            raw_data, vocabulary, seq_length, data_manager, eval_mode=False, logger=MockLogger(),
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            for_rnn=for_rnn, use_left_padding=use_left_padding)

        tokens, transitions, labels, num_transitions = data

        # Filter examples that don't have lengths <= seq_length
        assert tokens.shape == (7, seq_length)
        assert transitions.shape == (7, seq_length)

        for s, ts, num_t in zip(tokens, transitions, num_transitions):

            # The sentence should end with an operator.
            assert s_is_left_to_right(s, OPERATOR_TOKENS)

            # The sentences should be padded on the right.
            assert not s_is_left_padded(s)

            # The num_transitions should count non-skip transitions
            assert len([x for x in ts if x != util.SKIP_SYMBOL]) == num_t

            # The transitions should start with SKIP and end with REDUCE (ignoring SKIPs).
            assert t_is_left_to_right(ts)

            # The transitions should be padded on the left.
            assert t_is_left_padded(ts)


if __name__ == '__main__':
    unittest.main()
