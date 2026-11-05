#ifndef STABLEBEAR_TIMESERIES_POINT_CLOUDS_H
#define STABLEBEAR_TIMESERIES_POINT_CLOUDS_H

#include "../errors.hpp"
#include "../executor.hpp"
#include "../point_cloud.hpp"
#include "../tensor.hpp"
#include "../timeseries.hpp"
#include "../walk.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace sb
{
  /// The coordinate type of point clouds from series of @p T: float32 for
  /// float32, and float64 for other numbers.
  template <typename T>
  using point_coordinate_t = std::conditional_t<std::is_same_v<T, float32_t>, float32_t, float64_t>;

  /// The point cloud of @p series, one point per observation stored at a
  /// time in its domain, in the order of their times; a window with an open
  /// start leaves out the value it stores at its start. An observation of shape () gives a point of
  /// dimension 1, and one of shape (m) a point of dimension m. Requires one
  /// channel, and throws TypeError for observations of other shapes.
  template <typename T>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
  PointCloud<point_coordinate_t<T>> to_point_cloud(const TimeSeries<T>& series)
  {
    if (series.n_channels() != 1)
    {
      throw std::invalid_argument(
          "A point cloud comes from a series of one channel, not " + std::to_string(series.n_channels())
          + "; combine the channels with sb.stack or sb.concatenate, or select one with series[i]");
    }
    const auto& channel = series.channel(0);
    const auto& shape = channel.shape();
    if (shape.size() > 1)
    {
      throw TypeError("Point clouds come from observations of shape () or (m,), not of "
                      + std::to_string(shape.size()) + " dimensions");
    }
    const size_t dim = shape.empty() ? 1 : shape[0];
    std::vector<size_t> kept;
    if (channel.size() > 0)
    {
      const auto domain = series.domain();
      for (size_t k = 0; k < channel.size(); ++k)
      {
        if (domain.contains(channel.time(k)))
        {
          kept.push_back(k);
        }
      }
    }
    PointCloud<point_coordinate_t<T>> cloud({kept.size(), dim});
    for (size_t i = 0; i < kept.size(); ++i)
    {
      for (size_t j = 0; j < dim; ++j)
      {
        cloud(i, j) = static_cast<point_coordinate_t<T>>(channel.flat_value(kept[i] * dim + j));
      }
    }
    return cloud;
  }

  /// The point cloud of every series of @p seriesTensor, as to_point_cloud()
  /// gives it, in a tensor of the same shape. The clouds can differ in size.
  /// The series are converted in parallel on @p exec.
  template <typename T>
    requires(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>)
  Tensor<PointCloud<point_coordinate_t<T>>> to_point_clouds(const Tensor<TimeSeries<T>>& seriesTensor,
                                                            Executor& exec = default_executor())
  {
    Tensor<PointCloud<point_coordinate_t<T>>> clouds(
        std::vector<size_t>(seriesTensor.shape().begin(), seriesTensor.shape().end()));
    parallel_walk(seriesTensor, [&](const std::vector<size_t>& index) {
      clouds(index) = to_point_cloud(seriesTensor(index));
    }, exec);
    return clouds;
  }
}

#endif // STABLEBEAR_TIMESERIES_POINT_CLOUDS_H
