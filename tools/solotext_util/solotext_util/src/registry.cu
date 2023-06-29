#include <torch/script.h>
#include <torch/extension.h>

#include "clustering.h"
#include "mean.h"
#include "get_text_masks.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("clustering", &clustering, "clustering");
  m.def("ker_meaning", &meaning, "meaning");
  m.def("get_masks_counter", &get_masks_counter, "get_masks_counter");

}
