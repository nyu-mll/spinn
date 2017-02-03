import unittest
import argparse
import tempfile

from nose.plugins.attrib import attr
import numpy as np

import pytest

from spinn import util
from spinn.fat_stack import SPINN

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import spinn.util.blocks as blocks


class MockModel(nn.Module):
    def __init__(self, scalar=11):
        super(MockModel, self).__init__()
        self.layer = nn.Linear(2, 2)
        self.scalar = scalar


class PytorchTestCase(unittest.TestCase):

    def test_save_load(self):
        model = MockModel(11)
        model.child = MockModel(13)

        # Save to and load from temporary file.
        temp = tempfile.NamedTemporaryFile()
        torch.save(model, temp.name)
        _model = torch.load(temp.name)

        # Check length of parameters.
        assert len(list(model.parameters())) == 4
        assert len(list(model.parameters())) == len(list(_model.parameters()))

        # Check value of parameters.
        for w, _w in zip(model.parameters(), _model.parameters()):
            assert all((w.data == _w.data).numpy().astype(bool).tolist())

        # Check value of scalars.
        assert model.scalar == 11
        assert model.child.scalar == 13
        assert model.scalar == _model.scalar
        assert model.child.scalar == _model.child.scalar
        
        # Cleanup temporary file.
        temp.close()


class BlocksTestCase(unittest.TestCase):

    def test_expand_along(self):
        mock_rewards = np.array([1.0, 0.0, 2.0])
        mock_mask = np.array([[True, True], [False, True], [False, False]])
        ret = blocks.expand_along(mock_rewards, mock_mask)
        expected = [1., 1., 0.]
        assert len(ret) == len(expected)
        assert all(r == e for r, e in zip(ret, expected))

    def test_select_mask(self):
        t = torch.range(0, 9).view(-1, 2)
        var = Variable(t)
        mask = torch.ByteTensor([True, True, False, True, False])
        ret = blocks.select_mask(var, mask)
        expected = torch.cat([t[i].unsqueeze(0) for i, m in enumerate(mask) if m], 0)
        assert all(s1 == s2 for s1, s2 in zip(ret.view(-1).data, expected.view(-1)))
        assert ret.size() == expected.size()


if __name__ == '__main__':
    unittest.main()
