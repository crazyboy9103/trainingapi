import torch
import detectron2._C
print(dir(detectron2._C))

print(detectron2._C.get_compiler_version())
print(detectron2._C.get_cuda_version())
print(detectron2._C.has_cuda())

