"""
rnn_utils.py
------------
Utility functions for the RNN / LSTM From Scratch notebook.

Provides:
    - softmax(x)
    - sigmoid(x)
    - deepcopy helper (safe copy of the parameters dictionary)
"""

import numpy as np
import copy


# ---------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------
def softmax(x):
    """
    Compute the softmax of each column of x.

    Arguments:
    x -- numpy array of shape (n_y, m)

    Returns:
    y -- numpy array of shape (n_y, m), the softmax of each column
    """
    # Subtract max per column for numerical stability
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def sigmoid(x):
    """
    Compute the sigmoid of x (element-wise).

    Arguments:
    x -- scalar or numpy array of any size

    Returns:
    s -- sigmoid(x)
    """
    # Clip for numerical stability (avoid overflow in exp)
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------
# Copy helper
# ---------------------------------------------------------------------
def deepcopy_dict(d):
    """
    Return a deep copy of a dictionary (used for the parameters).
    """
    return copy.deepcopy(d)


# ---------------------------------------------------------------------
# Optional: a tiny helper to pretty-print parameter shapes
# (useful for debugging, not required by the notebook)
# ---------------------------------------------------------------------
def describe_parameters(parameters):
    """
    Print the name and shape of every entry in a parameters dict.
    """
    for k, v in parameters.items():
        if isinstance(v, np.ndarray):
            print(f"{k:5s} : shape = {v.shape}")
        else:
            print(f"{k:5s} : {type(v)}")
