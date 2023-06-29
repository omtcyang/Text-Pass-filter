#include <torch/script.h>
#include <torch/extension.h>

#include "get_text_masks.h"
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME solotext_util_ext
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_masks_counter", &get_masks_counter, "get_masks_counter");

}
