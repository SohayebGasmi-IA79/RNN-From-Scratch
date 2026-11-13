"""
public_tests.py
---------------
Unit tests for the RNN / LSTM From Scratch notebook.

Exposes:
    - rnn_cell_forward_tests(rnn_cell_forward)
    - rnn_forward_test(rnn_forward)
    - lstm_cell_forward_test(lstm_cell_forward)
    - lstm_forward_test(lstm_forward)
"""

import numpy as np


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------
def _assert_allclose(name, actual, expected, tol=1e-6):
    """Assert that two arrays are element-wise close; raise on failure."""
    actual   = np.asarray(actual,   dtype=float)
    expected = np.asarray(expected, dtype=float)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"[{name}] shape mismatch.\n"
            f"  expected shape: {expected.shape}\n"
            f"  obtained shape: {actual.shape}"
        )
    if not np.allclose(actual, expected, atol=tol, rtol=tol):
        raise AssertionError(
            f"[{name}] value mismatch.\n"
            f"  expected: {expected}\n"
            f"  obtained: {actual}\n"
            f"  diff    : {np.abs(actual - expected)}"
        )


def _print_green(msg):
    print(f"\033[92m{msg}\033[0m")


# ---------------------------------------------------------------------
# 1) RNN cell forward
# ---------------------------------------------------------------------
def rnn_cell_forward_tests(rnn_cell_forward):
    np.random.seed(1)
    xt      = np.random.randn(3, 10)
    a_prev  = np.random.randn(5, 10)
    parameters = {
        "Waa": np.random.randn(5, 5),
        "Wax": np.random.randn(5, 3),
        "Wya": np.random.randn(2, 5),
        "ba":  np.random.randn(5, 1),
        "by":  np.random.randn(2, 1),
    }

    a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)

    # ---- Shapes -------------------------------------------------------
    assert a_next.shape  == (5, 10), f"a_next.shape  = {a_next.shape}"
    assert yt_pred.shape == (2, 10), f"yt_pred.shape = {yt_pred.shape}"

    # ---- Values -------------------------------------------------------
    expected_a_next = np.array([
        0.59584544, 0.18141802, 0.61311866, 0.99808218, 0.85016201,
        0.99980978, -0.18887155, 0.99815551, 0.6531151, 0.82872037,
    ])
    expected_yt_pred = np.array([
        0.9888161, 0.01682021, 0.21140899, 0.36817467, 0.98988387,
        0.88945212, 0.36920224, 0.9966312, 0.9982559, 0.17746526,
    ])
    _assert_allclose("a_next[4]",  a_next[4],  expected_a_next)
    _assert_allclose("yt_pred[1]", yt_pred[1], expected_yt_pred)

    # ---- Cache --------------------------------------------------------
    assert len(cache) == 4, "cache must contain 4 elements"
    a_next_c, a_prev_c, xt_c, _ = cache
    _assert_allclose("cache[0] (a_next)", a_next_c, a_next)
    _assert_allclose("cache[1] (a_prev)", a_prev_c, a_prev)
    _assert_allclose("cache[2] (xt)",     xt_c,     xt)

    _print_green("All tests passed")


# ---------------------------------------------------------------------
# 2) RNN forward over Tx steps
# ---------------------------------------------------------------------
def rnn_forward_test(rnn_forward):
    np.random.seed(1)
    x  = np.random.randn(3, 10, 4)
    a0 = np.random.randn(5, 10)
    parameters = {
        "Waa": np.random.randn(5, 5),
        "Wax": np.random.randn(5, 3),
        "Wya": np.random.randn(2, 5),
        "ba":  np.random.randn(5, 1),
        "by":  np.random.randn(2, 1),
    }

    a, y_pred, caches = rnn_forward(x, a0, parameters)

    # ---- Shapes -------------------------------------------------------
    assert a.shape      == (5, 10, 4), f"a.shape      = {a.shape}"
    assert y_pred.shape == (2, 10, 4), f"y_pred.shape = {y_pred.shape}"

    # ---- Values -------------------------------------------------------
    _assert_allclose("a[4][1]",
                     a[4][1],
                     np.array([-0.99999375, 0.77911235,
                               -0.99861469, -0.99833267]))
    _assert_allclose("y_pred[1][3]",
                     y_pred[1][3],
                     np.array([0.79560373, 0.86224861,
                               0.11118257, 0.81515947]))

    # ---- Cache structure ---------------------------------------------
    assert isinstance(caches, tuple) and len(caches) == 2, \
        "caches must be a tuple (list_of_caches, x)"
    list_caches, x_cached = caches
    assert len(list_caches) == 4, "there must be Tx=4 cached entries"

    # >>> FIX: use caches[1][1][3]  (= x[1][3])  instead of
    #          list_caches[1][1][3] (= row 3 of a_prev, wrong shape)
    _assert_allclose("caches[1][1][3]",
                     caches[1][1][3],
                     np.array([-1.1425182, -0.34934272,
                               -0.20889423, 0.58662319]))

    _print_green("All tests passed")


# ---------------------------------------------------------------------
# 3) LSTM cell forward
# ---------------------------------------------------------------------
def lstm_cell_forward_test(lstm_cell_forward):
    np.random.seed(1)
    xt     = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    c_prev = np.random.randn(5, 10)
    parameters = {
        "Wf": np.random.randn(5, 5 + 3),
        "bf": np.random.randn(5, 1),
        "Wi": np.random.randn(5, 5 + 3),
        "bi": np.random.randn(5, 1),
        "Wo": np.random.randn(5, 5 + 3),
        "bo": np.random.randn(5, 1),
        "Wc": np.random.randn(5, 5 + 3),
        "bc": np.random.randn(5, 1),
        "Wy": np.random.randn(2, 5),
        "by": np.random.randn(2, 1),
    }

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    # ---- Shapes -------------------------------------------------------
    assert a_next.shape == (5, 10), f"a_next.shape = {a_next.shape}"
    assert c_next.shape == (5, 10), f"c_next.shape = {c_next.shape}"
    assert yt.shape     == (2, 10), f"yt.shape     = {yt.shape}"

    # ---- Values -------------------------------------------------------
    _assert_allclose("a_next[4]",
                     a_next[4],
                     np.array([-0.66408471, 0.0036921, 0.02088357, 0.22834167,
                               -0.85575339, 0.00138482, 0.76566531, 0.34631421,
                               -0.00215674, 0.43827275]))
    _assert_allclose("c_next[2]",
                     c_next[2],
                     np.array([0.63267805, 1.00570849, 0.35504474, 0.20690913,
                               -1.64566718, 0.11832942, 0.76449811, -0.0981561,
                               -0.74348425, -0.26810932]))
    _assert_allclose("yt[1]",
                     yt[1],
                     np.array([0.79913913, 0.15986619, 0.22412122, 0.15606108,
                               0.97057211, 0.31146381, 0.00943007, 0.12666353,
                               0.39380172, 0.07828381]))

    # ---- Cache length -------------------------------------------------
    assert len(cache) == 10, f"LSTM cache length = {len(cache)} (expected 10)"

    _print_green("All tests passed")


# ---------------------------------------------------------------------
# 4) LSTM forward over Tx steps
# ---------------------------------------------------------------------
def lstm_forward_test(lstm_forward):
    np.random.seed(1)
    x  = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    parameters = {
        "Wf": np.random.randn(5, 5 + 3),
        "bf": np.random.randn(5, 1),
        "Wi": np.random.randn(5, 5 + 3),
        "bi": np.random.randn(5, 1),
        "Wo": np.random.randn(5, 5 + 3),
        "bo": np.random.randn(5, 1),
        "Wc": np.random.randn(5, 5 + 3),
        "bc": np.random.randn(5, 1),
        "Wy": np.random.randn(2, 5),
        "by": np.random.randn(2, 1),
    }

    a, y, c, caches = lstm_forward(x, a0, parameters)

    # ---- Shapes -------------------------------------------------------
    assert a.shape == (5, 10, 7), f"a.shape = {a.shape}"
    assert y.shape == (2, 10, 7), f"y.shape = {y.shape}"
    assert c.shape == (5, 10, 7), f"c.shape = {c.shape}"

    # ---- Values -------------------------------------------------------
    _assert_allclose("a[4][3][6]", np.array(a[4][3][6]), np.array(0.172117767533))
    _assert_allclose("y[1][4][3]", np.array(y[1][4][3]), np.array(0.95087346185))

    _assert_allclose("caches[1][1][1]",
                     caches[1][1][1],
                     np.array([0.82797464, 0.23009474, 0.76201118,
                               -0.22232814, -0.20075807, 0.18656139,
                               0.41005165]))
    _assert_allclose("c[1][2][1]", np.array(c[1][2][1]), np.array(-0.855544916718))

    # ---- Cache structure ---------------------------------------------
    assert isinstance(caches, tuple) and len(caches) == 2, \
        "caches must be a tuple (list_of_caches, x)"
    list_caches, x_cached = caches
    assert len(list_caches) == 7, "there must be Tx=7 cached entries"

    _print_green("All tests passed")
