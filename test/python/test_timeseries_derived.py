import numpy as np
import pytest

from stablebear.timeseries import TimeDelayEmbedding, Times, TimeSeries


def _ms_series():
    # Each observation is 10 * t, so a value carried from an earlier
    # observation stands out: x(1.5) is 10, not 15.
    return TimeSeries([0.0, 10.0, 25.0, 40.0], times=Times([0.0, 1.0, 2.5, 4.0], unit="ms"))


def test_embedding_has_a_delayed_point_at_each_observation_time():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    # Points (x(t - 2), x(t - 1), x(t)) at 2.5 and 4.0, oldest first.
    assert np.array_equal(np.asarray(embedding.values), [[0.0, 10.0, 25.0], [10.0, 25.0, 40.0]])
    assert np.array_equal(np.asarray(embedding(3.25)), [0.0, 10.0, 25.0])


def test_embedding_starts_where_every_component_has_a_value():
    embedding = TimeDelayEmbedding(dimension=3, delay=1.0)(_ms_series())

    assert embedding.unit == "ms"
    assert embedding.shape == (3,)
    # 0.0 and 1.0 have no value two milliseconds earlier.
    assert np.array_equal(np.asarray(embedding.times), [2.5, 4.0])
    with pytest.raises(ValueError, match="outside the domain"):
        embedding(2.0)
