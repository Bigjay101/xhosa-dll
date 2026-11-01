"""P11: paired cluster resampling, sign-flip, Holm and ratio tests."""
import numpy as np
import pytest
from src.stats_extra import paired_cluster_bootstrap, sign_flip_test, holm, ratio_ci


def test_paired_constant():
    """Constant paired differences must have a zero-width confidence interval."""
    mu, lo, hi, reps = paired_cluster_bootstrap([2,2,2,2], ['a','a','b','b'], 100)
    assert (mu,lo,hi) == pytest.approx((2,2,2))
    assert np.all(reps==2)


def test_holm_and_ratio():
    """Holm values preserve input ordering; ratios use joint draws."""
    assert holm([.03,.01,.04]) == pytest.approx([.06,.03,.06])
    assert ratio_ci([1,2,3], [2,4,6]) == pytest.approx((.5,.5))


def test_cluster_sign_flip():
    """Symmetric pairs yield valid reproducible permutation p-values."""
    args=([1,-1,2,-2],['a','b','c','d'])
    first=sign_flip_test(*args,n_perm=100,seed=7)
    assert 0 <= first <= 1 and first == sign_flip_test(*args,n_perm=100,seed=7)
