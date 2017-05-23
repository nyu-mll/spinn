import torch
from torch.autograd import Variable


def auxiliary_loss(model):

    has_spinn = hasattr(model, 'spinn')
    has_policy = has_spinn and hasattr(model, 'policy_loss')
    has_value = has_spinn and hasattr(model, 'value_loss')

    total_loss = 0
    if has_policy:
        total_loss += model.policy_loss
    if has_value:
        total_loss += model.value_loss

    return total_loss

