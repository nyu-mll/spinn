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


from spinn.util.blocks import BaseSentencePairTrainer, Linear


class SentencePairTrainer(BaseSentencePairTrainer):

    def init_optimizer(self, lr=0.01, l2_lambda=0.0, **kwargs):
        relevant_params = [w for w in self.model.parameters() if w.requires_grad]
        self.optimizer = optim.SGD(relevant_params, lr=lr, weight_decay=l2_lambda)


class SentenceTrainer(SentencePairTrainer):
    pass


class BaseModel(nn.Module):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 initial_embeddings, num_classes, mlp_dim,
                 embedding_keep_rate, classifier_keep_rate,
                 use_tracker_dropout=True, tracker_dropout_rate=0.1,
                 use_input_dropout=False, use_input_norm=False,
                 use_classifier_norm=True,
                 gpu=-1,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 use_tracking_lstm=True,
                 use_tracking_in_composition=True,
                 use_shift_composition=True,
                 use_history=False,
                 save_stack=False,
                 use_reinforce=False,
                 projection_dim=None,
                 encoding_dim=None,
                 use_encode=False,
                 use_skips=False,
                 use_sentence_pair=False,
                 skip_embedding=False,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.skip_embedding = skip_embedding

        if not skip_embedding:
            if initial_embeddings is not None:
                self._embed = nn.Embedding(vocab_size, word_embedding_dim)
                self._embed.weight.data.set_(torch.from_numpy(initial_embeddings))
                self._embed.weight.requires_grad = False
            else:
                self._embed = nn.Embedding(vocab_size, word_embedding_dim)
                self._embed.weight.requires_grad = True

        # CBOW doesn't use model_dim right now. Let's leave this message here anyway for now, since
        # word_embedding_dim is effectively the model_dim.
        assert word_embedding_dim == model_dim, "Currently only supports word_embedding_dim == model_dim"

        mlp_input_dim = word_embedding_dim * 2 if use_sentence_pair else word_embedding_dim

        self.l0 = Linear(mlp_input_dim, mlp_dim)
        self.l1 = Linear(mlp_dim, mlp_dim)
        self.l2 = Linear(mlp_dim, num_classes)

        self.nonlinear = F.log_softmax if num_classes >= 2 else F.sigmoid


    def embed(self, x, train):
        return self._embed(x)


    def run_mlp(self, h, train):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = self.nonlinear(h)
        return y


class SentencePairModel(BaseModel):
    def forward(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        if not self.skip_embedding:
            # Build Tokens
            x_prem = Variable(sentences[:,:,0], volatile=not train)
            x_hyp = Variable(sentences[:,:,1], volatile=not train)
            x = torch.cat([x_prem, x_hyp], 0)

            emb = self.embed(x, train)
        else:
            emb = sentences

        batch_size = emb.size(0)

        hh = torch.squeeze(torch.sum(emb, dim=1))
        h = torch.cat([hh[:batch_size/2], hh[batch_size/2:]], 1)
        logits = self.run_mlp(h, train)

        if y_batch is not None:
            loss = F.nll_loss(logits, Variable(y_batch, volatile=not train))
            pred = logits.data.max(1)[1] # get the index of the max log-probability
            acc = pred.eq(y_batch).sum() / float(y_batch.size(0))
        else:
            loss = None
            acc = 0.0

        return logits, loss, acc, 0.0, None, None


class SentenceModel(BaseModel):
    def forward(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size, seq_length = sentences.size()[:2]

        if not self.skip_embedding:
            # Build Tokens
            x = Variable(sentences, volatile=not train)

            emb = self.embed(x, train)
        else:
            emb = sentences

        h = torch.squeeze(torch.sum(emb, dim=1)) / seq_length
        logits = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        if y_batch is not None:
            loss = F.nll_loss(logits, Variable(y_batch, volatile=not train))
            pred = logits.data.max(1)[1] # get the index of the max log-probability
            acc = pred.eq(y_batch).sum() / float(y_batch.size(0))
        else:
            loss = 0.0
            acc = 0.0

        return logits, loss, acc, 0.0, None, None
