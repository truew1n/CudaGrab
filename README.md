# CudaGrab

**CudaGrab** is a lightweight C++/CUDA DLL that enables high-performance GPU-side screen capturing and normalized preprocessing of a selected region.  
It provides direct access to a CUDA buffer, allowing fast integration into GPU or CPU pipelines.

## Features
- Capture screen regions directly into GPU memory (CUDA buffer).
- Preprocess (normalize) pixel data on the GPU.
- Extremely low latency by avoiding unnecessary CPU copies.
- Easy to call from Python, C++, or other languages.

## Quick Usage (Python Example)

```python
import ctypes
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import time

dll = ctypes.CDLL("./CudaGrab.dll")

dll.CreateContext.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
dll.CreateContext.restype = ctypes.c_ubyte
dll.CaptureScreen.restype = ctypes.c_ubyte
dll.PreprocessScreen.argtypes = [ctypes.c_float] * 6
dll.PreprocessScreen.restype = ctypes.c_ubyte
dll.GetMainBufferPointer.restype = ctypes.c_void_p
dll.CleanupDirect3DContext.restype = None

screen_width = 2560
screen_height = 1440
region_width = 600
region_height = 600

result = dll.CreateContext(screen_width, screen_height, region_width, region_height)
if result != 0:
    print(f"CreateDirect3DContext failed with code {result}")
    exit(1)

dll.CaptureScreen()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
result = dll.PreprocessScreen(*(mean + std))
if result != 0:
    print(f"PreprocessScreen failed with code {result}")
    dll.CleanupDirect3DContext()
    exit(3)

gpu_ptr = dll.GetMainBufferPointer()
buffer_size = region_width * region_height * 3
host_buffer = np.empty(buffer_size, dtype=np.float32)
cuda.memcpy_dtoh(host_buffer, gpu_ptr)

image_np = host_buffer.reshape((3, region_height, region_width)).transpose(1, 2, 0)
image_uint8 = np.clip((image_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(image_uint8).save("preprocessed_capture.png")

dll.CleanupContext()
```

## Exposed DLL Functions

| Function | Description |
| -------- | ----------- |
| `CreateContext(width, height, region_width, region_height)` | Initializes D3D11 and CUDA context. |
| `CaptureScreen()` | Captures the screen into a GPU texture. |
| `PreprocessScreen(meanR, meanG, meanB, stdR, stdG, stdB)` | Normalizes the captured image. |
| `GetMainBufferPointer()` | Returns a pointer to the image buffer data (normalized if Preprocess has been called). |
| `CleanupContext()` | Releases all allocated resources. |

## Requirements
- Windows with Direct3D 11 support
- CUDA Toolkit (with CUDA D3D11 Interop)
- Python packages: pycuda, numpy, Pillow
