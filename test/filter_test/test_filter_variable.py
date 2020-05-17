import numpy as np
import matplotlib.pyplot as plt
from hpe3d.filter import filter_variable


def test_filter_variable():

    # Test constant position filtering mode
    x1 = np.ones((100, 1), dtype=float)
    x1_filt = filter_variable(x1, mode='c')
    np.testing.assert_allclose(x1, x1_filt)

    # Test constant velocity filtering mode (linear position)
    x2 = np.arange(100, dtype=float)[:, np.newaxis]
    x2_filt = filter_variable(x2, mode='v')
    np.testing.assert_allclose(x2, x2_filt, rtol=0.5)

    # Test constant acceleration filtering mode (quadratic position)
    x3 = x2 ** 2
    x3_filt = filter_variable(x3, mode='a')
    np.testing.assert_allclose(x3, x3_filt, rtol=8.)

    # Test dimensionality
    n_dim = np.random.randint(1,20)
    x4 = np.ones((200, n_dim), dtype=float)
    x4_filt = filter_variable(x4, mode='c')
    assert x4.shape == x4_filt.shape

    x4_filt = filter_variable(x4, mode='v')
    assert x4.shape == x4_filt.shape

    x4_filt = filter_variable(x4, mode='a')
    assert x4.shape == x4_filt.shape

    print('test_filter_variable:\tsuccessful!')


test_filter_variable()
