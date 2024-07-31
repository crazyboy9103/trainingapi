// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>
#include "roi_align_rotated/ROIAlignRotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "nms_rotated/nms_rotated.h"

namespace detectron2 {

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream oss;

#if defined(WITH_CUDA)
  oss << "CUDA ";
#else
  oss << "HIP ";
#endif
  return oss.str();
#else // neither CUDA nor HIP
  return std::string("not available");
#endif
}

bool has_cuda() {
#if defined(WITH_CUDA)
  return true;
#else
  return false;
#endif
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // helpers
  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");
  m.def("has_cuda", &has_cuda, "has_cuda");
  // ops
  m.def("nms_rotated", &nms_rotated);
  m.def("box_iou_rotated", &box_iou_rotated);
  m.def("roi_align_rotated_forward", &ROIAlignRotated_forward);
  m.def("roi_align_rotated_backward", &ROIAlignRotated_backward);
}
} // namespace detectron2