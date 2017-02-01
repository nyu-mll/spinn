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


from spinn.util.blocks import BaseSentencePairTrainer, HeKaimingInit


class SentencePairTrainer(BaseSentencePairTrainer):

    def init_optimizer(self, lr=0.01, l2_lambda=0.0, **kwargs):
        relevant_params = [w for w in self.model.parameters() if w.requires_grad]
        self.optimizer = optim.Adam(relevant_params, lr=lr, betas=(0.9, 0.999), weight_decay=l2_lambda)


class SentenceTrainer(SentencePairTrainer):
    pass


class BaseModel(nn.Module):
    def __init__(self, model_dim, word_embedding_dim, vocab_size,
                 seq_length, initial_embeddings, num_classes, mlp_dim,
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
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.model_dim = model_dim
        self.use_encode = use_encode
        self.bi_encode = False

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

        if use_encode:
            self.encode = nn.LSTM(word_embedding_dim, model_dim, 1,
                # batch_first=True,
                bidirectional=self.bi_encode,
                )

        self.rnn = nn.LSTM(word_embedding_dim, model_dim, 1)

        mlp_input_dim = word_embedding_dim * 2 if use_sentence_pair else model_dim
        
        self.l0 = nn.Linear(mlp_input_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)

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

    def embed(self, x, train):
        return self._embed(x)

    def run_rnn(self, x, train):
        batch_size, seq_len, model_dim = x.data.size()

        num_layers = 1
        bidirectional = False
        bi = 2 if bidirectional else 1
        h0 = Variable(torch.zeros(num_layers * bi, batch_size, self.model_dim), volatile=not train)
        c0 = Variable(torch.zeros(num_layers * bi, batch_size, self.model_dim), volatile=not train)

        # Transforms x
        #   from  => batch_size x seq_len x model_dim
        #   to    => seq_len x batch_size x model_dim
        x = torch.transpose(x, 0, 1)

        # Expects (input, h_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.rnn(x, (h0, c0))

        return hn

    def run_encode(self, x, train):
        batch_size, seq_len, model_dim = x.data.size()

        num_layers = 1
        bidirectional = self.bi_encode
        bi = 2 if bidirectional else 1
        h0 = Variable(torch.zeros(num_layers * bi, batch_size, self.model_dim), volatile=not train)
        c0 = Variable(torch.zeros(num_layers * bi, batch_size, self.model_dim), volatile=not train)

        # Transforms x
        #   from  => batch_size x seq_len x model_dim
        #   to    => seq_len x batch_size x model_dim
        x = torch.transpose(x, 0, 1)

        # Expects (input, h_0):
        #   input => seq_len x batch_size x model_dim
        #   h_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.encode(x, (h0, c0))
        output = torch.transpose(output, 0, 1)

        return output

    def run_mlp(self, h, train):
        h = self.l0(h)
        h = F.relu(h)
        h = self.l1(h)
        h = F.relu(h)
        h = self.l2(h)
        y = F.log_softmax(h)
        return y


class SentencePairModel(BaseModel):
    def __call__(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size, seq_length = sentences.size()[:2]

        # Build Tokens
        x_prem = Variable(sentences[:,:,0], volatile=not train)
        x_hyp = Variable(sentences[:,:,1], volatile=not train)
        x = torch.cat([x_prem, x_hyp], 0)

        emb = self.embed(x, train)

        if self.use_encode:
            emb = self.run_encode(emb, train)

        hh = torch.squeeze(self.run_rnn(emb, train))
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        logits = self.run_mlp(h, train)

        if y_batch is not None:
            loss = F.nll_loss(logits, Variable(y_batch, volatile=not train))
            pred = logits.data.max(1)[1] # get the index of the max log-probability
            acc = pred.eq(y_batch).sum() / float(y_batch.size(0))
        return logits, loss, acc, 0.0, None, None


class SentenceModel(BaseModel):
    def __call__(self, sentences, transitions, y_batch=None, train=True, **kwargs):
        batch_size, seq_length = sentences.size()

        # Build Tokens
        x = Variable(sentences, volatile=not train)

        emb = self.embed(x, train)

        if self.use_encode:
            emb = self.run_encode(emb, train)

        h = torch.squeeze(self.run_rnn(emb, train))
        logits = self.run_mlp(h, train)

        _, h, _ = self.fwd_rnn(embeds, train, keep_hs=False)
        y = self.run_mlp(h, train)

        # Calculate Loss & Accuracy.
        if y_batch is not None:
            loss = F.nll_loss(logits, Variable(y_batch, volatile=not train))
            pred = logits.data.max(1)[1] # get the index of the max log-probability
            acc = pred.eq(y_batch).sum() / float(y_batch.size(0))
        else:
            accum_loss = 0.0
            acc = 0.0

        return logits, loss, acc, 0.0, None, None
