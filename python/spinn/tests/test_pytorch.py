import unittest

import tempfile

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


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


if __name__ == '__main__':
    unittest.main()
