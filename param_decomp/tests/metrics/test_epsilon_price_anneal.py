import pytest

from param_decomp.metrics.decoder_usage_price import annealed_epsilon


def test_none_final_is_constant_peak():
    assert annealed_epsilon(1e-2, None, 0.0, 1.0, 0.5) == 1e-2


def test_endpoints_held_flat_outside_window():
    assert annealed_epsilon(1e-2, 1e-6, 0.25, 0.75, 0.1) == 1e-2  # before window
    assert annealed_epsilon(1e-2, 1e-6, 0.25, 0.75, 0.9) == 1e-6  # after window


def test_loglinear_midpoint_is_geometric_mean():
    # peak=1e-2, final=1e-6 → log-linear midpoint = sqrt(1e-2 * 1e-6) = 1e-4
    assert annealed_epsilon(1e-2, 1e-6, 0.0, 1.0, 0.5) == pytest.approx(1e-4)


def test_monotone_decreasing_across_window():
    vals = [annealed_epsilon(1e-2, 1e-6, 0.0, 1.0, f) for f in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


def test_positive_endpoints_asserted():
    with pytest.raises(AssertionError):
        annealed_epsilon(1e-2, 0.0, 0.0, 1.0, 0.5)
