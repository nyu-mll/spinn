"""Dataset handling and related yuck."""

import random
import itertools
import time
import sys

import numpy as np


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"

TRANSITION_PADDING_SYMBOL = 2
SENTENCE_PADDING_SYMBOL = 0

CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}

# Allowed number of transition types : currently PUSH : 0 and MERGE : 1
NUM_TRANSITION_TYPES = 2


def TrimDataset(dataset, seq_length, eval_mode=False, sentence_pair_data=False):
    """Avoid using excessively long training examples."""
    if eval_mode:
        return dataset
    else:
        if sentence_pair_data:
            new_dataset = [example for example in dataset if
                len(example["premise_transitions"]) <= seq_length and
                len(example["hypothesis_transitions"]) <= seq_length]
        else:
            new_dataset = [example for example in dataset if len(
                example["transitions"]) <= seq_length]
        return new_dataset


def TokensToIDs(vocabulary, dataset, sentence_pair_data=False):
    """Replace strings in original boolean dataset with token IDs."""
    if sentence_pair_data:
        keys = ["premise_tokens", "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for key in keys:
        if UNK_TOKEN in vocabulary:
            unk_id = vocabulary[UNK_TOKEN]
            for example in dataset:
                example[key] = [vocabulary.get(token, unk_id)
                                     for token in example[key]]
        else:
            for example in dataset:
                example[key] = [vocabulary[token]
                                for token in example[key]]
    return dataset


def CropAndPadExample(example, left_padding, target_length, key, symbol=0, logger=None):
    """
    Crop/pad a sequence value of the given dict `example`.
    """
    if left_padding < 0:
        # Crop, then pad normally.
        # TODO: Track how many sentences are cropped, but don't log a message
        # for every single one.
        example[key] = example[key][-left_padding:]
        left_padding = 0
    right_padding = target_length - (left_padding + len(example[key]))
    example[key] = ([symbol] * left_padding) + \
        example[key] + ([symbol] * right_padding)


def CropAndPad(dataset, length, logger=None, sentence_pair_data=False):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    # Always make sure that the transitions are aligned at the left edge, so
    # the final stack top is the root of the tree. If cropping is used, it should
    # just introduce empty nodes into the tree.
    if sentence_pair_data:
        keys = [("premise_transitions", "num_premise_transitions", "premise_tokens"),
                ("hypothesis_transitions", "num_hypothesis_transitions", "hypothesis_tokens")]
    else:
        keys = [("transitions", "num_transitions", "tokens")]

    for example in dataset:
        for (transitions_key, num_transitions_key, tokens_key) in keys:
            example[num_transitions_key] = len(example[transitions_key])
            transitions_left_padding = 0 # length - example[num_transitions_key]
            shifts_before_crop_and_pad = example[transitions_key].count(0)
            CropAndPadExample(
                example, transitions_left_padding, length, transitions_key,
                symbol=TRANSITION_PADDING_SYMBOL, logger=logger)
            shifts_after_crop_and_pad = example[transitions_key].count(0)
            tokens_left_padding = shifts_after_crop_and_pad - \
                shifts_before_crop_and_pad
            CropAndPadExample(
                example, tokens_left_padding, length, tokens_key,
                symbol=SENTENCE_PADDING_SYMBOL, logger=logger)
    return dataset

def CropAndPadForRNN(dataset, length, logger=None, sentence_pair_data=False):
    # NOTE: This can probably be done faster in NumPy if it winds up making a
    # difference.
    if sentence_pair_data:
        keys = ["premise_tokens",
                "hypothesis_tokens"]
    else:
        keys = ["tokens"]

    for example in dataset:
        for tokens_key in keys:
            num_tokens = len(example[tokens_key])
            tokens_left_padding = length - num_tokens
            CropAndPadExample(
                example, tokens_left_padding, length, tokens_key,
                symbol=SENTENCE_PADDING_SYMBOL, logger=logger)
    return dataset


def merge(x, y):
    return ''.join([a for t in zip(x, y) for a in t])


def peano(x, y):
    interim = ''.join(merge(format(x, '08b'), format(y, '08b')))
    return int(interim, base=2)


def MakeTrainingIterator(sources, batch_size, smart_batches=True, use_peano=True):
    # Make an iterator that exposes a dataset as random minibatches.

    def get_key(num_transitions):
        if use_peano:
            prem_len, hyp_len = num_transitions
            key = peano(prem_len, hyp_len)
            return key
        else:
            if not isinstance(num_transitions, list):
                num_transitions = [num_transitions]
            return max(num_transitions)

    def build_batches():
        dataset_size = len(sources[0])
        seq_length = sources[0].shape[1]
        order = range(dataset_size)
        random.shuffle(order)
        order = np.array(order)

        num_splits = 10 # TODO: Should we be smarter about split size?
        order_limit = len(order) / num_splits * num_splits
        order = order[:order_limit]
        order_splits = np.split(order, num_splits)
        batches = []

        for split in order_splits:
            # Put indices into buckets based on example length.
            keys = []
            for i in split:
                num_transitions = sources[3][i]
                key = get_key(num_transitions)
                keys.append((i, key))
            keys = sorted(keys, key=lambda (_, key): key)

            # Group indices from buckets into batches, so that
            # examples in each batch have similar length.
            batch = []
            for i, _ in keys:
                batch.append(i)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch = []
            if len(batch) > 0:
                print "WARNING: May be discarding {} train examples.".format(len(batch))
        return batches

    def batch_iter():
        batches = build_batches()
        num_batches = len(batches)
        idx = -1
        order = range(num_batches)
        random.shuffle(order)

        while True:
            idx += 1
            if idx >= num_batches:
                # Start another epoch.
                batches = build_batches()
                num_batches = len(batches)
                idx = 0
                order = range(num_batches)
                random.shuffle(order)
            batch_indices = batches[order[idx]]
            yield tuple(source[batch_indices] for source in sources)

    def data_iter():
        dataset_size = len(sources[0])
        start = -1 * batch_size
        order = range(dataset_size)
        random.shuffle(order)

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            yield tuple(source[batch_indices] for source in sources)

    train_iter = batch_iter if smart_batches else data_iter

    return train_iter()


def MakeEvalIterator(sources, batch_size, limit=-1):
    # Make a list of minibatches from a dataset to use as an iterator.
    # TODO(SB): Pad out the last few examples in the eval set if they don't
    # form a batch.

    print "WARNING: May be discarding eval examples."

    dataset_size = limit if limit >= 0 else len(sources[0])
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size

        if start >= dataset_size:
            break

        candidate_batch = tuple(source[start:start + batch_size]
                               for source in sources)

        if len(candidate_batch[0]) == batch_size:
            data_iter.append(candidate_batch)
        else:
            print "Skipping " + str(len(candidate_batch[0])) + " examples."
    return data_iter


def PreprocessDataset(dataset, vocabulary, seq_length, data_manager, eval_mode=False, logger=None,
                      sentence_pair_data=False, for_rnn=False):
    # TODO(SB): Simpler version for plain RNN.
    dataset = TrimDataset(dataset, seq_length, eval_mode=eval_mode, sentence_pair_data=sentence_pair_data)
    dataset = TokensToIDs(vocabulary, dataset, sentence_pair_data=sentence_pair_data)
    if for_rnn:
        dataset = CropAndPadForRNN(dataset, seq_length, logger=logger, sentence_pair_data=sentence_pair_data)
    else:
        dataset = CropAndPad(dataset, seq_length, logger=logger, sentence_pair_data=sentence_pair_data)

    if sentence_pair_data:
        X = np.transpose(np.array([[example["premise_tokens"] for example in dataset],
                      [example["hypothesis_tokens"] for example in dataset]],
                     dtype=np.int32), (1, 2, 0))
        if for_rnn:
            # TODO(SB): Extend this clause to the non-pair case.
            transitions = np.zeros((len(dataset), 2, 0))
            num_transitions = np.transpose(np.array(
                [[len(np.array(example["premise_tokens"]).nonzero()[0]) for example in dataset],
                 [len(np.array(example["hypothesis_tokens"]).nonzero()[0]) for example in dataset]],
                dtype=np.int32), (1, 0))
        else:
            transitions = np.transpose(np.array([[example["premise_transitions"] for example in dataset],
                                    [example["hypothesis_transitions"] for example in dataset]],
                                   dtype=np.int32), (1, 2, 0))
            num_transitions = np.transpose(np.array(
                [[example["num_premise_transitions"] for example in dataset],
                 [example["num_hypothesis_transitions"] for example in dataset]],
                dtype=np.int32), (1, 0))
    else:
        X = np.array([example["tokens"] for example in dataset],
                     dtype=np.int32)
        transitions = np.array([example["transitions"] for example in dataset],
                               dtype=np.int32)
        num_transitions = np.array(
            [example["num_transitions"] for example in dataset],
            dtype=np.int32)
    y = np.array(
        [data_manager.LABEL_MAP[example["label"]] for example in dataset],
        dtype=np.int32)

    return X, transitions, y, num_transitions


def BuildVocabulary(raw_training_data, raw_eval_sets, embedding_path, logger=None, sentence_pair_data=False):
    # Find the set of words that occur in the data.
    logger.Log("Constructing vocabulary...")
    types_in_data = set()
    for dataset in [raw_training_data] + [eval_dataset[1] for eval_dataset in raw_eval_sets]:
        if sentence_pair_data:
            types_in_data.update(itertools.chain.from_iterable([example["premise_tokens"]
                                                                for example in dataset]))
            types_in_data.update(itertools.chain.from_iterable([example["hypothesis_tokens"]
                                                                for example in dataset]))
        else:
            types_in_data.update(itertools.chain.from_iterable([example["tokens"]
                                                                for example in dataset]))
    logger.Log("Found " + str(len(types_in_data)) + " word types.")

    if embedding_path == None:
        logger.Log(
            "Warning: Open-vocabulary models require pretrained vectors. Running with empty vocabulary.")
        vocabulary = CORE_VOCABULARY
    else:
        # Build a vocabulary of words in the data for which we have an
        # embedding.
        vocabulary = BuildVocabularyForASCIIEmbeddingFile(
            embedding_path, types_in_data, CORE_VOCABULARY)

    return vocabulary


def BuildVocabularyForASCIIEmbeddingFile(path, types_in_data, core_vocabulary):
    """Quickly iterates through a GloVe-formatted ASCII vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    # TODO(SB): Report on *which* words are skipped. See if any are common.

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ", 1)
            word = spl[0]
            if word in types_in_data:
                vocabulary[word] = next_index
                next_index += 1
    return vocabulary


def LoadEmbeddingsFromASCII(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format ASCII vector file.

    For now, values not found in the file will be set to zero."""
    emb = np.zeros(
        (len(vocabulary), embedding_dim), dtype=np.float32)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:]]
    return emb


def TransitionsToParse(transitions, words):
    if transitions is not None:
        stack = ["(P *ZEROS*)"] * (len(transitions) + 1)
        buffer_ptr = 0
        for transition in transitions:
            if transition == 0:
                stack.append("(P " + words[buffer_ptr] +")")
                buffer_ptr += 1
            elif transition == 1:
                r = stack.pop()
                l = stack.pop()
                stack.append("(M " + l + " " + r + ")")
        return stack.pop()
    else:
        return " ".join(words)


class SimpleProgressBar(object):
    """ Simple Progress Bar and Timing Snippet
    """

    def __init__(self, msg=">", bar_length=80, enabled=True):
        super(SimpleProgressBar, self).__init__()
        self.enabled = enabled
        if not self.enabled: return

        self.begin = time.time()
        self.bar_length = bar_length
        self.msg = msg

    def step(self, i, total):
        if not self.enabled: return
        sys.stdout.write('\r')
        pct = (i / float(total)) * 100
        ii = i * self.bar_length / total
        fmt = "%s [%-{}s] %d%% %ds / %ds".format(self.bar_length)
        total_time = time.time()-self.begin
        expected = total_time / ((i+1e-03) / float(total))
        sys.stdout.write(fmt % (self.msg, '='*ii, pct, total_time, expected))
        sys.stdout.flush()

    def reset(self):
        if not self.enabled: return
        self.begin = time.time()

    def finish(self):
        if not self.enabled: return
        self.reset()
        sys.stdout.write('\n')
