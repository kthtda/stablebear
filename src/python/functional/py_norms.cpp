#include "py_distance.hpp"

#include "tensor.hpp"
#include "functional/pcf.hpp"
#include "task.hpp"
#include "../py_async_support.hpp"
#include "algorithms/functional/apply_functional.hpp"

#include <memory>

namespace nb = nanobind;

namespace
{

  template <typename Tt, typename Tv>
  class PyNormsBindings
  {
  public:
    using PcfT = sb::Pcf<Tt, Tv>;
    using TensorT = sb::Tensor<PcfT>;
    using OutTensorT = sb::Tensor<Tv>;

    template <typename F>
    static nb::tuple apply(TensorT inTensor, F f)
    {
      auto outTensor = OutTensorT(inTensor.shape(), Tv(0));
      std::unique_ptr<sb::StoppableTask<void>> task = sb_py::execute_stoppable_task<sb::ApplyFunctional<TensorT, OutTensorT, F>>(inTensor, outTensor, f);
      return nb::make_tuple(std::move(task), outTensor);
    }

    static nb::tuple lpnorm_l1(TensorT inTensor)
    {
      return apply(inTensor, [](const PcfT& v) { return sb::l1_norm(v); });
    }

    /// The gradient of each PCF's lp norm with respect to its breakpoint values,
    /// as PCFs with the same times that hold the gradients as values.
    static nb::tuple lpnorm_lp_grad_values(TensorT inTensor, Tv p)
    {
      auto outTensor = TensorT(inTensor.shape());
      auto f = [p](const PcfT& v) {
        auto pts = v.points();
        auto grad = sb::lp_norm_grad_values(v, p);
        for (size_t k = 0; k < pts.size(); ++k)
        {
          pts[k].v = grad[k];
        }
        return PcfT(std::move(pts));
      };
      std::unique_ptr<sb::StoppableTask<void>> task = sb_py::execute_stoppable_task<sb::ApplyFunctional<TensorT, TensorT, decltype(f)>>(inTensor, outTensor, f);
      return nb::make_tuple(std::move(task), outTensor);
    }

    /// Binds name(tensor, p, derivative=False), which applies f(pcf, p) to
    /// every PCF, or its derivative with respect to p if derivative is true.
    /// f must accept p as Tv or as sb::Dual<Tv>.
    template <typename F>
    static void def_static_wrt(nb::class_<PyNormsBindings>& cls, const char* name, F f)
    {
      cls.def_static(name, [f](TensorT inTensor, Tv p, bool derivative) {
        if (derivative)
        {
          return apply(inTensor, [p, f](const PcfT& v) { return f(v, sb::Dual<Tv>(p, 1)).d; });
        }
        return apply(inTensor, [p, f](const PcfT& v) { return f(v, p); });
      }, nb::arg("tensor"), nb::arg("p"), nb::arg("derivative") = false);
    }

    static void register_bindings(nb::handle m, const std::string& suffix)
    {
      nb::class_<PyNormsBindings> cls(m, ("Norms" + suffix).c_str());

      cls.def_static("lpnorm_l1", &PyNormsBindings::lpnorm_l1);
      cls.def_static("lpnorm_lp_grad_values", &PyNormsBindings::lpnorm_lp_grad_values);
      def_static_wrt(cls, "lpnorm_lp", [](const PcfT& v, auto q) { return sb::lp_norm(v, q); });
    }
  };

}

namespace sb_py
{

  void register_norms(nb::module_& m)
  {
    PyNormsBindings<sb::float32_t, sb::float32_t>::register_bindings(m, "_f32_f32");
    PyNormsBindings<sb::float64_t, sb::float64_t>::register_bindings(m, "_f64_f64");
  }

}