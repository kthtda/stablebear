#ifndef STABLEBEAR_BARCODE_SUMMARY_H
#define STABLEBEAR_BARCODE_SUMMARY_H

#include "../ufunc.hpp"
#include "../functional/pcf.hpp"
#include "barcode.hpp"

#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sb::ph
{
  /**
   * A barcode-to-PCF transformation as an operation without extra dimensions:
   * each barcode maps to one PCF.
   *
   * @tparam T       Scalar type (e.g. float32_t, float64_t)
   * @tparam Func    A callable taking a const Barcode<T>&, such as one
   *                 returning Pcf<T,T>
   */
  template <typename T, typename Func>
  struct BarcodeSummaryOperation
  {
    using input_type = Barcode<T>;
    using output_type = std::invoke_result_t<const Func&, const Barcode<T>&>;

    Func func;

    [[nodiscard]] std::vector<size_t> extra_dims() const { return {}; }

    void operator()(const Barcode<T>& barcode, std::span<output_type> out) const
    {
      out[0] = func(barcode);
    }
  };

  template <typename B>
  struct barcode_scalar;

  template <typename T>
  struct barcode_scalar<Barcode<T>>
  {
    using type = T;
  };

  /// The scalar type of the barcodes in a container, such as a tensor or a
  /// time series of barcodes.
  template <typename Container>
  using barcode_scalar_t = typename barcode_scalar<typename Container::value_type>::type;

  /// Apply @p func to every barcode in @p barcodes, a tensor or a time
  /// series, in parallel.
  template <typename Container, typename Func>
  auto make_barcode_summary_task(Container barcodes, Func func, std::string description)
  {
    using T = barcode_scalar_t<Container>;
    return make_ufunc_task(
        std::move(barcodes), BarcodeSummaryOperation<T, Func>{std::move(func)},
        std::move(description), "barcode");
  }

} // namespace sb::ph

#endif // STABLEBEAR_BARCODE_SUMMARY_H
