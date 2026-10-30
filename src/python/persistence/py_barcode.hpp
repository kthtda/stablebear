#ifndef STABLEBEAR_PY_BARCODE_H
#define STABLEBEAR_PY_BARCODE_H

#include <sbear/persistence/persistence_pair.hpp>

#include "../nb.hpp"

namespace sb_py
{
  void register_persistence_barcode_tensor(nb::module_& m);
}

#endif //STABLEBEAR_PY_BARCODE_H