import numpy as np

_loss_functions = {}
def get_loss_function(name, **kwargs):
    return _loss_functions[name](**kwargs)

def _register_name(name):
    def _registerer(cls):
        _loss_functions[name] = cls
        return cls
    return _registerer

class LossFunctionBase(object):
    def __init__(self, margin=0):
        self.margin = margin


@_register_name("relu")
class ReluLoss(LossFunctionBase):

    def compute_loss(self, d, rx, ry, label):
        energy = d - rx + ry
        return np.maximum(0, np.where(label, energy, -energy))
        
    def compute_loss_diff(self, d, rx, ry, label):
        energy = d - rx + ry

        cond = ((energy >= 0) == label.astype(bool)).astype(float)
        sign = np.where(label, 1, -1)
        ret = cond*sign

        return ret, -ret, ret

@_register_name("relu_margin")
class ReluMarginLoss(LossFunctionBase):

    def compute_loss(self, d, rx, ry, label):
        energy = d - rx + ry
        return np.maximum(0, np.where(label, energy, self.margin - energy))
        
    def compute_loss_diff(self, d, rx, ry, label):
        energy = d - rx + ry
        ret = np.where(label,
                1.0 * (energy>0),
                -1.0 * (energy<=self.margin)
                )
        return ret, -ret, ret

@_register_name("margin")
class MarginLoss(LossFunctionBase):

    def compute_loss(self, d, rx, ry, label):
        energy = np.maximum(0, d - rx + ry)

        return np.where(label, np.maximum(0, energy), np.maximum(0, self.margin-energy))
        
    def compute_loss_diff(self, d, rx, ry, label):
        energy = d - rx + ry
        ret = np.where(label,
                1.0,
                -(energy<=self.margin).astype(float),
                ) * (energy>0)
        return ret, -ret, ret

@_register_name("margin_sq")
class MarginSquareLoss(LossFunctionBase):

    def compute_loss(self, d, rx, ry, label):
        energy = np.maximum(0, d - rx + ry)

        return np.where(label, np.maximum(0, energy), np.maximum(0, self.margin-energy))**2
        
    def compute_loss_diff(self, d, rx, ry, label):
        energy = d - rx + ry
        ret = np.where(label,
                2*np.maximum(0, energy),
                -2*np.maximum(0, 0.5-energy)*(energy>0)
                )

        return ret, -ret, ret

