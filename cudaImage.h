//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <cuda_runtime.h>

class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
  cudaStream_t stream;
public:
  CudaImage();
  ~CudaImage();
  CudaImage(CudaImage &&other) noexcept;
  CudaImage(const CudaImage &other) = delete;
  CudaImage &operator=(CudaImage &&other) noexcept;
  CudaImage &operator=(const CudaImage &other) = delete;

  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL, cudaStream_t stream = 0);
  double Download();
  double Readback();
  double InitTexture();
  double CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);
void StartTimer(unsigned int *hTimer);
double StopTimer(unsigned int hTimer);

#endif // CUDAIMAGE_H
