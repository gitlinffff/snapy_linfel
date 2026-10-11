#pragma once

// torch
#include <ATen/TensorIterator.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

using ideal_gas_fn = void (*)(at::TensorIterator &iter, double gammad);
// count: the caller reads the failure count (on CUDA a device-to-host read)
using fix_vapor_fn = int (*)(at::TensorIterator &iter, bool count);

DECLARE_DISPATCH(ideal_gas_fn, ideal_gas_cons2prim);
DECLARE_DISPATCH(fix_vapor_fn, call_fix_vapor);

}  // namespace at::native
