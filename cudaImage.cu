//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include "cudautils.h"
#include "cudaImage.h"

#include <cstdio>

int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem, cudaStream_t str)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  stream = str;
  if (devmem==NULL) {
    safeCall(cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height));
    pitch /= sizeof(float);
    if (d_data==NULL)
      printf("Failed to allocate device data\n");
    d_internalAlloc = true;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() :
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL)
    safeCall(cudaFree(d_data));
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL)
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL)
    safeCall(cudaFreeArray((cudaArray *)t_data));
  t_data = NULL;
}

double CudaImage::Download()
{
  TimerGPU timer(stream);
  auto p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL)
    safeCall(cudaMemcpy2DAsync(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice, stream));
//  safeCall(cudaStreamSynchronize(stream));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Download time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::Readback()
{
  TimerGPU timer(stream);
  auto p = sizeof(float)*pitch;
  safeCall(cudaMemcpy2DAsync(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost, stream));
//  safeCall(cudaStreamSynchronize(stream));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("Readback time =               %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::InitTexture()
{
  TimerGPU timer(stream);
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  safeCall(cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height));
  if (t_data==NULL)
    printf("Failed to allocated texture data\n");
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("InitTexture time =            %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

double CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    printf("Error CopyToTexture: No texture data\n");
    return 0.0;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    printf("Error CopyToTexture: No source data\n");
    return 0.0;
  }
  TimerGPU timer(stream);
  if (host)
    safeCall(cudaMemcpy2DToArrayAsync((cudaArray *)dst.t_data, 0, 0, h_data,
             sizeof(*h_data)*pitch, sizeof(*h_data)*pitch, dst.height, cudaMemcpyHostToDevice, stream));
  else
    safeCall(cudaMemcpy2DToArrayAsync((cudaArray *)dst.t_data, 0, 0, d_data,
             sizeof(*h_data)*pitch, sizeof(*h_data)*pitch, dst.height, cudaMemcpyDeviceToDevice, stream));
//  safeCall(cudaStreamSynchronize(stream));
  double gpuTime = timer.read();
#ifdef VERBOSE
  printf("CopyToTexture time =          %.2f ms\n", gpuTime);
#endif
  return gpuTime;
}

CudaImage::CudaImage(CudaImage &&other) noexcept :
    width(other.width), height(other.height), pitch(other.pitch),
    h_data(other.h_data), d_data(other.d_data), t_data(other.t_data),
    d_internalAlloc(other.d_internalAlloc), h_internalAlloc(other.h_internalAlloc),
    stream(other.stream) {
  other.h_data = nullptr;
  other.d_data = nullptr;
  other.t_data = nullptr;
}

CudaImage &CudaImage::operator=(CudaImage &&other) noexcept {
  if (&other == this)
    return *this;

  width = other.width;
  height = other.height;
  pitch = other.pitch;
  h_data = other.h_data;
  d_data = other.d_data;
  t_data = other.t_data;
  d_internalAlloc = other.d_internalAlloc;
  h_internalAlloc = other.h_internalAlloc;
  stream = other.stream;

  other.h_data = nullptr;
  other.d_data = nullptr;
  other.t_data = nullptr;
  return *this;
}
