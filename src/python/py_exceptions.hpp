#ifndef STABLEBEAR_PY_EXCEPTIONS_H
#define STABLEBEAR_PY_EXCEPTIONS_H

#include "nb.hpp"

#include <exception>
#include <string>

namespace sb_py
{
  /// Add the exception type @p name, a subclass of @p base, to @p m, and
  /// raise it for every C++ exception of type @p T, with the message of @p T
  /// and the attributes that @p setAttributes(error, e) sets on it. As for
  /// nb::exception, which sets only the message, the type lives as long as
  /// the interpreter.
  template <typename T, auto setAttributes>
  void register_exception(nb::module_& m, const char* name, const char* doc, nb::handle base)
  {
    const auto qualified = nb::cast<std::string>(m.attr("__name__")) + "." + name;
    PyObject* type = PyErr_NewExceptionWithDoc(qualified.c_str(), doc, base.ptr(), nullptr);
    if (!type)
    {
      throw nb::python_error();
    }
    m.attr(name) = nb::handle(type);

    nb::register_exception_translator(
        [](const std::exception_ptr& p, void* payload) {
          try
          {
            std::rethrow_exception(p);
          }
          catch (const T& e)
          {
            const nb::handle exceptionType(static_cast<PyObject*>(payload));
            nb::object error = exceptionType(e.what());
            setAttributes(error, e);
            PyErr_SetObject(exceptionType.ptr(), error.ptr());
          }
        },
        type);
  }
}

#endif // STABLEBEAR_PY_EXCEPTIONS_H
