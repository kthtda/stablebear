"""Static type checks for the public API.

pyright checks this file in CI; it is not run. Each ``assert_type`` fails
the check if the inferred type of the expression changes.
"""

import numpy as np
from typing_extensions import assert_type

import stablebear as sb
from stablebear.timeseries import Domain, OutsideDomainError, TimeSeries, cut_windows, to_point_clouds
from stablebear.timeseries.date_time import DateTime, DateTimeTensor, TimeDelta, TimeDeltaTensor

pcfs = sb.zeros((3,))

assert_type(sb.max_time(pcfs), sb.FloatTensor)
assert_type(sb.mean(pcfs), sb.PcfTensor)
assert_type(sb.sum(pcfs), sb.PcfTensor)

assert_type(DateTime("2023-01-01").count, int)
assert_type(DateTime("2023-01-01").to_datetime64(), np.datetime64)
assert_type(DateTimeTensor(["2023-01-01"]).count, sb.IntTensor)
assert_type(DateTimeTensor(["2023-01-01"]).to_datetime64(), np.ndarray)

assert_type(TimeDelta(1, "h").count, int | float)
assert_type(TimeDelta(1, "h").to_timedelta64(), np.timedelta64)
assert_type(TimeDeltaTensor([1], "h").count, sb.IntTensor | sb.FloatTensor)
assert_type(TimeDeltaTensor([1], "h").to_timedelta64(), np.ndarray)

outside = OutsideDomainError("Time 3 is outside the domain [0, 1]", 3.0, Domain(0.0, 1.0, True, True))

assert_type(outside.time, DateTime | TimeDelta | float)
assert_type(outside.domain, Domain)
assert_type(outside.index, tuple[int, ...] | None)

cloud = sb.PointCloud(np.zeros((4, 2)))
assert_type(sb.random.subsample(cloud, n_points=2), sb.PointCloudTensor)
assert_type(sb.random.subsample(sb.DistanceMatrix(4), n_points=2), sb.DistanceMatrixTensor)

assert_type(sb.distributions.Gaussian().weight(0.5), float)
assert_type(sb.distributions.Gaussian().weight(np.zeros(3)), np.ndarray)

hour = TimeDelta(1, "h")
hours = TimeDeltaTensor([1, 2], "h")
assert_type(hour + (30, "min"), TimeDelta)
assert_type(hour + hours, TimeDeltaTensor)
assert_type(hours - hour, TimeDeltaTensor)
assert_type(-hour, TimeDelta)
assert_type(hours * 0.5, TimeDeltaTensor)
assert_type(hour < hour, bool)
assert_type(hours < hour, sb.BoolTensor)
assert_type(hour == hours, sb.BoolTensor)
assert_type(hour == hour, bool)

day = DateTime("2023-01-01")
days = DateTimeTensor(["2023-01-01", "2023-01-02"])
assert_type(day + hour, DateTime)
assert_type(day + hours, DateTimeTensor)
assert_type(day - day, TimeDelta)
assert_type(days - day, TimeDeltaTensor)
assert_type(days < day, sb.BoolTensor)
assert_type(day < day, bool)

observed = TimeSeries([10.0, 20.0], times=hours)
assert_type(to_point_clouds(observed), sb.PointCloud)
assert_type(to_point_clouds(cut_windows(observed, size=hour, step=hour)), sb.PointCloudTensor)
assert_type(sb.stack(observed), TimeSeries)
assert_type(sb.concatenate(observed), TimeSeries)
