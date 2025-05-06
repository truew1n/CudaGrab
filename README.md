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

dll.CreateContext.argtypes = [ctypes.c_int, ctypes.c_int]
dll.CreateContext.restype = ctypes.c_ubyte

dll.CaptureScreen.argtypes = []
dll.CaptureScreen.restype = ctypes.c_ubyte

dll.PreprocessScreen.argtypes = []
dll.PreprocessScreen.restype = ctypes.c_ubyte

dll.GetMainBufferPointer.argtypes = []
dll.GetMainBufferPointer.restype = ctypes.c_void_p

dll.CleanupContext.argtypes = []
dll.CleanupContext.restype = None

region_width = 600
region_height = 600
num_channels = 3

result = dll.CreateContext(region_width, region_height)
if result != 0:
    print(f"CreateContext failed with code {result}")
    exit(1)
print("Direct3D and CUDA context created.")

start_time = time.perf_counter()

result = dll.CaptureScreen()
if result != 0:
    print(f"CaptureScreen failed with code {result}")
    dll.CleanupContext()
    exit(2)
print("Screen captured.")

result = dll.PreprocessScreen()
if result != 0:
    print(f"PreprocessScreen failed with code {result}")
    dll.CleanupContext()
    exit(3)
print("Preprocessing complete.")

end_time = time.perf_counter()
print(f"Total time (Capture + Preprocess): {end_time - start_time:.4f} seconds")

gpu_ptr = dll.GetMainBufferPointer()
if not gpu_ptr:
    print("Failed to get GPU buffer pointer.")
    dll.CleanupContext()
    exit(4)

buffer_size = region_width * region_height * num_channels
host_buffer = np.empty(buffer_size, dtype=np.float32)

cuda.memcpy_dtoh(host_buffer, gpu_ptr)
image_np = host_buffer.reshape((3, region_height, region_width)).transpose(1, 2, 0)
image_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
Image.fromarray(image_uint8).save("preprocessed_capture.png")
print("Saved as 'preprocessed_capture.png'")

dll.CleanupContext()
print("Cleanup done.")
```

## Exposed DLL Functions

| Function | Description |
| -------- | ----------- |
| `CreateContext(region_width, region_height)` | Initializes D3D11 and CUDA context. |
| `CaptureScreen()` | Captures the screen into a GPU texture. |
| `PreprocessScreen()` | Normalizes the captured image. |
| `GetMainBufferPointer()` | Returns a pointer to the image buffer data. |
| `CleanupContext()` | Releases all allocated resources. |

## Requirements
- Windows with Direct3D 11 support
- CUDA Toolkit (with CUDA D3D11 Interop)
- Python packages: pycuda, numpy, Pillow
