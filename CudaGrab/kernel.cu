#include <d3d11.h>
#include <dxgi1_2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_d3d11_interop.h>
#include <Windows.h>
#include <cstdio>

static ID3D11Device *G_D3DDevice = nullptr;
static ID3D11DeviceContext *G_D3DContext = nullptr;
static cudaGraphicsResource *G_CudaResource = nullptr;
static ID3D11Texture2D *G_ScreenTexture = nullptr;
static IDXGIOutputDuplication *G_OutputDuplication = nullptr;

static int G_Width = 2560;
static int G_Height = 1440;
static int G_RegionWidth = 600;
static int G_RegionHeight = 600;

static unsigned char *G_TempBuffer = nullptr;
static float *G_MainBuffer = nullptr;

__global__ void Preprocess(
    float *Out, unsigned char *In, int Width, int Height,
    float MeanR, float MeanG, float MeanB,
    float StdR, float StdG, float StdB
)
{
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;
    if (X >= Width || Y >= Height) return;

    int InIdx = (Y * Width + X) * 4;
    int OutIdxR = (0 * Height + Y) * Width + X;
    int OutIdxG = (1 * Height + Y) * Width + X;
    int OutIdxB = (2 * Height + Y) * Width + X;

    Out[OutIdxR] = ((float)In[InIdx + 2] / 255.0f - MeanR) / StdR;
    Out[OutIdxG] = ((float)In[InIdx + 1] / 255.0f - MeanG) / StdG;
    Out[OutIdxB] = ((float)In[InIdx + 0] / 255.0f - MeanB) / StdB;
}

extern "C" __declspec(dllexport) void CleanupContext() {
    if (G_CudaResource) {
        cudaGraphicsUnmapResources(1, &G_CudaResource, 0);
        cudaGraphicsUnregisterResource(G_CudaResource);
        G_CudaResource = nullptr;
    }
    if (G_TempBuffer) {
        cudaFree(G_TempBuffer);
        G_TempBuffer = nullptr;
    }
    if (G_MainBuffer) {
        cudaFree(G_MainBuffer);
        G_MainBuffer = nullptr;
    }
    if (G_ScreenTexture) {
        G_ScreenTexture->Release();
        G_ScreenTexture = nullptr;
    }
    if (G_OutputDuplication) {
        G_OutputDuplication->Release();
        G_OutputDuplication = nullptr;
    }
    if (G_D3DContext) {
        G_D3DContext->Release();
        G_D3DContext = nullptr;
    }
    if (G_D3DDevice) {
        G_D3DDevice->Release();
        G_D3DDevice = nullptr;
    }
}

extern "C" __declspec(dllexport) BYTE CreateContext(int Width, int Height, int RegionWidth, int RegionHeight)
{
    G_Width = Width;
    G_Height = Height;
    G_RegionWidth = RegionWidth;
    G_RegionHeight = RegionHeight;

    if (G_D3DDevice != nullptr && G_D3DContext != nullptr && G_ScreenTexture != nullptr && G_OutputDuplication != nullptr) {
        return 0;
    }

    HRESULT HResult = S_OK;
    D3D_FEATURE_LEVEL FeatureLevel;

    HResult = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &G_D3DDevice, &FeatureLevel, &G_D3DContext
    );
    if (FAILED(HResult)) {
        printf("Failed to create Direct3D device and context: %ld\n", HResult);
        return 1;
    }

    cudaError_t CudaStatus = cudaD3D11SetDirect3DDevice(G_D3DDevice);
    if (CudaStatus != cudaSuccess) {
        printf("Failed to set CUDA Direct3D device: %s\n", cudaGetErrorString(CudaStatus));
        CleanupContext();
        return 2;
    }

    IDXGIFactory1 *DxgiFactory = nullptr;
    HResult = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&DxgiFactory);
    if (FAILED(HResult)) {
        printf("Failed to create DXGI factory: %ld\n", HResult);
        CleanupContext();
        return 3;
    }

    IDXGIAdapter *Adapter = nullptr;
    HResult = DxgiFactory->EnumAdapters(0, &Adapter);
    if (FAILED(HResult)) {
        printf("Failed to enumerate adapters: %ld\n", HResult);
        DxgiFactory->Release();
        CleanupContext();
        return 4;
    }

    IDXGIOutput *Output = nullptr;
    HResult = Adapter->EnumOutputs(0, &Output);
    if (FAILED(HResult)) {
        printf("Failed to enumerate outputs: %ld\n", HResult);
        Adapter->Release();
        DxgiFactory->Release();
        CleanupContext();
        return 5;
    }

    IDXGIOutput1 *Output1 = nullptr;
    HResult = Output->QueryInterface(__uuidof(IDXGIOutput1), (void **)&Output1);
    Output->Release();
    if (FAILED(HResult)) {
        printf("Failed to get IDXGIOutput1: %ld\n", HResult);
        Adapter->Release();
        DxgiFactory->Release();
        CleanupContext();
        return 6;
    }

    HResult = Output1->DuplicateOutput(G_D3DDevice, &G_OutputDuplication);
    Output1->Release();
    Adapter->Release();
    DxgiFactory->Release();
    if (FAILED(HResult)) {
        printf("Failed to create output duplication: %ld\n", HResult);
        CleanupContext();
        return 7;
    }

    D3D11_TEXTURE2D_DESC TextureDesc = {};
    TextureDesc.Width = G_RegionWidth;
    TextureDesc.Height = G_RegionHeight;
    TextureDesc.MipLevels = 1;
    TextureDesc.ArraySize = 1;
    TextureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    TextureDesc.SampleDesc.Count = 1;
    TextureDesc.Usage = D3D11_USAGE_DEFAULT;
    TextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    HResult = G_D3DDevice->CreateTexture2D(&TextureDesc, nullptr, &G_ScreenTexture);
    if (FAILED(HResult)) {
        printf("Failed to create texture for region: %ld\n", HResult);
        CleanupContext();
        return 8;
    }

    CudaStatus = cudaGraphicsD3D11RegisterResource(&G_CudaResource, G_ScreenTexture, cudaGraphicsRegisterFlagsNone);
    if (CudaStatus != cudaSuccess) {
        printf("Failed to register D3D resource with CUDA: %s\n", cudaGetErrorString(CudaStatus));
        CleanupContext();
        return 9;
    }

    CudaStatus = cudaGraphicsMapResources(1, &G_CudaResource, 0);
    if (CudaStatus != cudaSuccess) {
        printf("Failed to map CUDA resource: %s\n", cudaGetErrorString(CudaStatus));
        CleanupContext();
        return 10;
    }

    CudaStatus = cudaMalloc(&G_TempBuffer, G_RegionWidth * G_RegionHeight * 4 * sizeof(unsigned int));
    if (CudaStatus != cudaSuccess) {
        printf("Failed to allocate temporary buffer: %s\n", cudaGetErrorString(CudaStatus));
        return 11;
    }

    CudaStatus = cudaMalloc(&G_MainBuffer, 3 * G_RegionWidth * G_RegionHeight * sizeof(float));
    if (CudaStatus != cudaSuccess) {
        printf("Failed to allocate CUDA main buffer: %s\n", cudaGetErrorString(CudaStatus));
        CleanupContext();
        return 12;
    }

    if (!G_D3DDevice || !G_D3DContext || !G_ScreenTexture || !G_OutputDuplication || !G_CudaResource || !G_TempBuffer || !G_MainBuffer) {
        printf("Incomplete initialization: Device=%p, Context=%p, Texture=%p, OutputDuplication=%p, CudaResource=%p, TempBuffer=%p, MainBuffer=%p\n",
            G_D3DDevice, G_D3DContext, G_ScreenTexture, G_OutputDuplication, G_CudaResource, G_TempBuffer, G_MainBuffer);
        CleanupContext();
        return 13;
    }

    return 0;
}

extern "C" __declspec(dllexport) BYTE CaptureScreen()
{
    if (G_D3DDevice == nullptr) {
        printf("D3D Device is null\n");
        return 1;
    }
    if (G_D3DContext == nullptr) {
        printf("D3D Context is null\n");
        return 2;
    }
    if (G_ScreenTexture == nullptr) {
        printf("Screen Texture is null\n");
        return 3;
    }
    if (G_OutputDuplication == nullptr) {
        printf("Output Duplication is null\n");
        return 4;
    }

    IDXGIResource *DesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO FrameInfo;
    HRESULT HResult = G_OutputDuplication->AcquireNextFrame(100, &FrameInfo, &DesktopResource);
    if (FAILED(HResult)) {
        printf("Failed to acquire next frame: %ld\n", HResult);
        return 5;
    }

    ID3D11Texture2D *DesktopTexture = nullptr;
    HResult = DesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&DesktopTexture);
    DesktopResource->Release();
    if (FAILED(HResult)) {
        printf("Failed to get desktop texture: %ld\n", HResult);
        G_OutputDuplication->ReleaseFrame();
        return 6;
    }

    int StartX = (G_Width - G_RegionWidth) / 2;
    int StartY = (G_Height - G_RegionHeight) / 2;

    D3D11_BOX Box;
    Box.left = StartX;
    Box.top = StartY;
    Box.right = StartX + G_RegionWidth;
    Box.bottom = StartY + G_RegionHeight;
    Box.front = 0;
    Box.back = 1;
    G_D3DContext->CopySubresourceRegion(G_ScreenTexture, 0, 0, 0, 0, DesktopTexture, 0, &Box);

    DesktopTexture->Release();
    G_OutputDuplication->ReleaseFrame();

    return 0;
}

extern "C" __declspec(dllexport) BYTE PreprocessScreen(
    float MeanR, float MeanG, float MeanB,
    float StdR, float StdG, float StdB
)
{
    if (G_CudaResource == nullptr || G_TempBuffer == nullptr || G_MainBuffer == nullptr) {
        printf("CUDA resource, temp buffer, or main buffer not initialized\n");
        return 1;
    }

    cudaArray_t CudaArray;
    cudaError_t CudaStatus = cudaGraphicsSubResourceGetMappedArray(&CudaArray, G_CudaResource, 0, 0);
    if (CudaStatus != cudaSuccess) {
        printf("Failed to get mapped CUDA array: %s\n", cudaGetErrorString(CudaStatus));
        return 2;
    }

    cudaMemcpy2DFromArray(
        G_TempBuffer,
        G_RegionWidth * 4,
        CudaArray,
        0, 0,
        G_RegionWidth * 4,
        G_RegionHeight,
        cudaMemcpyDeviceToDevice
    );

    dim3 BlockDim(16, 16);
    dim3 GridDim((G_RegionWidth + BlockDim.x - 1) / BlockDim.x, (G_RegionHeight + BlockDim.y - 1) / BlockDim.y);

    Preprocess<<<GridDim, BlockDim>>>(
        G_MainBuffer, G_TempBuffer, G_RegionWidth, G_RegionHeight,
        MeanR, MeanG, MeanB,
        StdR, StdG, StdB
    );

    CudaStatus = cudaGetLastError();
    if (CudaStatus != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(CudaStatus));
        return 4;
    }

    CudaStatus = cudaDeviceSynchronize();
    if (CudaStatus != cudaSuccess) {
        printf("CUDA device synchronization failed: %s\n", cudaGetErrorString(CudaStatus));
        return 5;
    }

    return 0;
}

extern "C" __declspec(dllexport) float *GetMainBufferPointer()
{
    return G_MainBuffer;
}
