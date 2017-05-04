import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import LSTMState, Embed, MLP, Linear, LSTM
from spinn.util.blocks import reverse_tensor
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.misc import Args, Vocab, Example

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP

from spinn.core.recursive import SPINN
from spinn.core.attention import SequenceAttention


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS, context_args, composition_args):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         use_sentence_pair=use_sentence_pair,
         lateral_tracking=FLAGS.lateral_tracking,
         use_tracking_in_composition=FLAGS.use_tracking_in_composition,
         predict_use_cell=FLAGS.predict_use_cell,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         use_attention=FLAGS.use_attention,
         attention_dim=FLAGS.attention_dim,
         mlp_dim=FLAGS.mlp_dim,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         context_args=context_args,
         composition_args=composition_args,
        )


class BaseModel(nn.Module):

    optimize_transition_loss = True

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 use_attention=None,
                 attention_dim=None,
                 lateral_tracking=None,
                 use_tracking_in_composition=None,
                 predict_use_cell=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 classifier_keep_rate=None,
                 context_args=None,
                 composition_args=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature

        self.hidden_dim = composition_args.size
        self.wrap_items = composition_args.wrap_items
        self.extract_h = composition_args.extract_h

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build parsing component.
        self.spinn = self.build_spinn(composition_args, vocab, predict_use_cell)

        self.use_attention = use_attention
        self.attention_dim = attention_dim
        if self.use_attention:
            self.attention = SequenceAttention(self.hidden_dim, self.hidden_dim, self.attention_dim)

        # Build classiifer.
        features_dim = self.get_features_dim()
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

        self.embedding_dropout_rate = 1. - embedding_keep_rate

        # Create dynamic embedding layer.
        self.embed = Embed(word_embedding_dim, vocab.size, vectors=vocab.vectors)

        self.input_dim = context_args.input_dim

        self.encode = context_args.encoder
        self.reshape_input = context_args.reshape_input
        self.reshape_context = context_args.reshape_context

    def get_features_dim(self):
        features_dim = self.hidden_dim * 2 if self.use_sentence_pair else self.hidden_dim
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += self.hidden_dim
            if self.use_product_feature:
                features_dim += self.hidden_dim
        return features_dim

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            features = torch.cat(features, 1)
        else:
            features = h[0]
        return features

    def build_spinn(self, args, vocab, predict_use_cell):
        return SPINN(args, vocab, predict_use_cell)

    def run_spinn(self, example, use_internal_parser, validate_transitions=True):
        self.spinn.reset_state()
        h_list, transition_acc, transition_loss = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        h = self.wrap(h_list)
        return h, transition_acc, transition_loss

    def forward_hook(self, embeds, batch_size, seq_length):
        pass

    def output_hook(self, output, sentences, transitions, y_batch=None):
        pass

    def forward(self, sentences, transitions, y_batch=None,
                 use_internal_parser=False, validate_transitions=True):
        example = self.unwrap(sentences, transitions)

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = self.reshape_input(embeds, b, l)
        embeds = self.encode(embeds)
        embeds = self.reshape_context(embeds, b, l)
        self.forward_hook(embeds, b, l)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)

        # Make Buffers
        ee = torch.chunk(embeds, b * l, 0)[::-1]
        bb = []
        for ii in range(b):
            ex = list(ee[ii*l:(ii+1)*l])
            bb.append(ex)
        buffers = bb[::-1]

        example.bufs = buffers

        h, transition_acc, transition_loss = self.run_spinn(example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        features = h

        # Run attention
        if self.use_attention:
            features = self.attend(features)

        # Build features
        features = self.build_features(features)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, h_list):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(h_list)
        return self.wrap_sentence(h_list)

    def attend(self, h_list):
        if self.use_sentence_pair:
            return self.attend_sentence_pair(h_list)
        return self.attend_sentence(h_list)

    # --- Sentence Model Specific ---

    def unwrap_sentence(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence(self, items):
        batch_size = len(items) / 2
        h = self.extract_h(self.wrap_items(items))
        return [h]

    def attend_sentence(self, items):
        batch_size, model_dim = items[0].size()

        root = items[0]
        nodes = map(lambda x: x['top_stack_1'].h, self.spinn.memories)
        sequence = torch.cat(nodes + [root], 1).view(batch_size, -1, model_dim)
        query = items[0]

        new_h, alphas = self.attention(sequence, query)

        return [new_h]

    # --- Sentence Pair Model Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence_pair(self, items):
        batch_size = len(items) / 2
        h_premise = self.extract_h(self.wrap_items(items[:batch_size]))
        h_hypothesis = self.extract_h(self.wrap_items(items[batch_size:]))
        return [h_premise, h_hypothesis]
