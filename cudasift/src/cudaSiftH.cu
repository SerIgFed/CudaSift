//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "cudasift/cudautils.h"
#include "cudasift/cudaImage.h"
#include "cudasift/cudaSift.h"
#include "cudasift/cudaSiftD.h"
#include "cudasift/cudaSiftH.h"

#include "cudaSiftD.cu"

void InitCuda(int maxPts, int numOctaves, float initBlur, int devNum)
{
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  if (!nDevices) {
    std::cerr << "No CUDA devices available" << std::endl;
    return;
  }
  devNum = std::min(nDevices-1, devNum);
  deviceInit(devNum);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  printf("Device Number: %d\n", devNum);
  printf("  Device name: %s\n", prop.name);
  printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate/1000);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %.1f\n\n",
     2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  printf("Initializing constant memory. \n"
         "This operation shouldn't be performed while SIFT extraction "
         "is running.\n");
  float variance = 0.5f;
  float scale = max(initBlur, 0.001f);
  {
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &maxPts,
                                sizeof(int), 0, cudaMemcpyHostToDevice));
  }
  {
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbol(
        d_LaplaceKernel, kernel, sizeof(kernel), 0,
        cudaMemcpyHostToDevice));
  }
  {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;
    safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, sizeof(h_Kernel),
                                0, cudaMemcpyHostToDevice));
  }
  {
    float kernel[2*LOWPASS_R+1];
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale);
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
      kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
      kernelSum += kernel[j+LOWPASS_R];
    }
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++)
      kernel[j+LOWPASS_R] /= kernelSum;
    safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, sizeof(kernel),
                                0, cudaMemcpyHostToDevice));
  }
}

template <typename T>
void forOctaves(int width, int height, int num_octaves, T &&cb) {
  for (int i = 0; i <= num_octaves; ++i) {
    const int p = iAlignUp(width, 128);
    if (!cb(i, width, height, p))
      return;
    width /= 2;
    height /= 2;
  }
}

void TempMemory::setSize(int w, int h) {
  restrict_width = w;
  restrict_height = h;
}

CudaImage TempMemory::image(int octave, cudaStream_t stream) const {
  CudaImage subImg;
  float *img_offset = imageBuffer();
  int rw = restrict_width, rh = restrict_height;
  forOctaves(width, height, num_octaves,
             [&](int i, int, int h, int p) {
               if (i == num_octaves - octave) {
                 subImg.Allocate(rw, rh, p, false, img_offset, nullptr, stream);
                 return false;
               }
               img_offset += h * p;
               rw /= 2;
               rh /= 2;
               return true;
             });
  return subImg;
}

cudaTextureObject_t TempMemory::texture(int octave) const {
  return textures[num_octaves - octave];
}

TempMemory::TempMemory(int width_, int height_, int num_octaves_, bool scale_up)
    : width( width_ *(scale_up ? 2 : 1)),
      height(height_*(scale_up ? 2 : 1)),
      restrict_width(width), restrict_height(height),
      num_octaves(num_octaves_) {
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
  const int nd = NUM_SCALES + 3;
  size_t images_size = 0;
  laplace_buffer_size = 0;
  forOctaves(width, height, num_octaves,
      [this, &images_size](int, int, int h, int p) {
    images_size += h*p;
    laplace_buffer_size += nd*h*p;
    return true;
  });
  size_t pitch;
  const size_t size = images_size + laplace_buffer_size;
  safeCall(cudaMallocPitch((void **)&d_data, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float)));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif

  float *img_offset = imageBuffer();
  forOctaves(width, height, num_octaves,
             [this, &img_offset](int i, int w, int h, int p) {
    if (i == num_octaves)
      return false;
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img_offset;
    resDesc.res.pitch2D.width = w;
    resDesc.res.pitch2D.height = h;
    resDesc.res.pitch2D.pitchInBytes = p * sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    textures.push_back(texObj);
    img_offset += h*p;
    return true;
  });

  d_PointCounter = nullptr;
  safeCall(cudaMalloc(&d_PointCounter, (8*2+1)*sizeof(*d_PointCounter)));
}

TempMemory::TempMemory(TempMemory &&other) noexcept
    : textures(std::move(other.textures)), d_data(other.d_data),
      laplace_buffer_size(other.laplace_buffer_size),
      width(other.width), height(other.height),
      restrict_width(other.restrict_width), restrict_height(other.restrict_height),
      num_octaves(other.num_octaves),
      d_PointCounter(other.d_PointCounter) {
  other.d_PointCounter = nullptr;
  other.d_data = nullptr;
}

TempMemory &TempMemory::operator=(TempMemory &&other) noexcept {
  if (this == &other)
    return *this;
  this->~TempMemory();

  textures = std::move(other.textures);
  d_data = other.d_data;
  laplace_buffer_size = other.laplace_buffer_size;
  width = other.width;
  height = other.height;
  restrict_width = other.restrict_width;
  restrict_height = other.restrict_height;
  num_octaves = other.num_octaves;
  other.d_data = nullptr;
  d_PointCounter = other.d_PointCounter;
  other.d_PointCounter = nullptr;
  return *this;
}

TempMemory::~TempMemory() {
  for (auto tex : textures)
    safeCall(cudaDestroyTextureObject(tex));
  safeCall(cudaFree(d_data));
  safeCall(cudaFree(d_PointCounter));
}

void ExtractSift(DeviceSiftData &siftData,
                 const DeviceDescriptorNormalizerData &d_normalizer,
                 const CudaImage &img, int numOctaves, float thresh,
                 float lowestScale, bool scaleUp, TempMemory &tempMemory,
                 cudaStream_t stream) {
//  TimerGPU timer(stream);
  safeCall(cudaMemsetAsync(tempMemory.pointCounter(), 0, (8*2+1)*sizeof(int), stream));

  int width = img.width*(scaleUp ? 2 : 1);
  int height = img.height*(scaleUp ? 2 : 1);

  CudaImage lowImg = tempMemory.image(numOctaves, stream);
  if (!scaleUp) {
    LowPass(lowImg, img, stream);
//    TimerGPU timer1(stream);
    ExtractSiftLoop(siftData, lowImg, d_normalizer, numOctaves, 0.0f, thresh, lowestScale, 1.0f, tempMemory, stream);
    safeCall(cudaMemcpyAsync(&siftData.numPts, &tempMemory.pointCounter()[2*numOctaves],
             sizeof(int), cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
//    printf("SIFT extraction time =        %.2f ms %d\n", timer1.read(), siftData.numPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, lowImg.pitch, false, tempMemory.laplaceBuffer(), nullptr, stream);
//    TimerGPU timer1(stream);
    ScaleUp(upImg, img, stream);
    LowPass(lowImg, upImg, stream);
    ExtractSiftLoop(siftData, lowImg, d_normalizer, numOctaves, 0.0f, thresh, lowestScale*2.0f,
                    1.0f, tempMemory, stream);
    safeCall(cudaMemcpyAsync(&siftData.numPts, &tempMemory.pointCounter()[2*numOctaves],
                             sizeof(int), cudaMemcpyDeviceToHost, stream));
    safeCall(cudaStreamSynchronize(stream));
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    if (siftData.numPts > 0)
      RescalePositions(siftData, 0.5f, stream);
//    printf("SIFT extraction time =        %.2f ms\n", timer1.read());
  }
//  double totTime = timer.read();
//  printf("Incl prefiltering & memcpy =  %.2f ms %d\n\n", totTime, siftData.numPts);
}

int ExtractSiftLoop(DeviceSiftData &siftData, const CudaImage &img,
                    const DeviceDescriptorNormalizerData &d_normalizer,
                    int numOctaves, double initBlur, float thresh, float lowestScale,
                    float subsampling, TempMemory &memoryTmp, cudaStream_t stream)
{
#ifdef VERBOSE
  TimerGPU timer(stream);
#endif
  if (numOctaves>1) {
    CudaImage subImg = memoryTmp.image(numOctaves - 1, stream);
    ScaleDown(subImg, img, stream);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, d_normalizer, numOctaves-1, totInitBlur, thresh,
                    lowestScale, subsampling*2.0f, memoryTmp, stream);
  }
  ExtractSiftOctave(siftData, img, d_normalizer, numOctaves, thresh, lowestScale,
                    subsampling, memoryTmp, stream);
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
  return 0;
}

void ExtractSiftOctave(DeviceSiftData &siftData, const CudaImage &img,
                       const DeviceDescriptorNormalizerData &d_normalizer,
                       int octave, float thresh, float lowestScale,
                       float subsampling, TempMemory &memoryTmp,
                       cudaStream_t stream)
{
  const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
  safeCall(cudaGetSymbolAddress((void**)&tempMemory.pointCounter(), d_PointCounter));
  unsigned int fstPts, totPts;
  safeCall(cudaMemcpy(&fstPts, &tempMemory.pointCounter()[2*octave-1], sizeof(int), cudaMemcpyDeviceToHost));
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = img.pitch;
  for (int i=0;i<nd-1;i++) {
    diffImg[i].Allocate(w, h, p, false, memoryTmp.laplaceBuffer() + i * p * h,
                        nullptr, stream);
  }

  auto texObj = memoryTmp.texture(octave);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(img, diffImg, octave, stream);
  FindPointsMulti(diffImg, siftData, memoryTmp, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave, stream);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
  ComputeOrientations(texObj, siftData, memoryTmp, octave, stream);
  ExtractSiftDescriptors(texObj, siftData, memoryTmp, d_normalizer, subsampling, octave, stream);
  //OrientAndExtract(texObj, siftData, subsampling, octave, stream);
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpy(&totPts, &tempMemory.pointCounter()[2*octave+1], sizeof(int), cudaMemcpyDeviceToHost));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0)
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts);
#endif
}

void PrintSiftData(SiftData &data)
{
  SiftPoint *h_data = data.h_data;
  for (int i=0;i<data.numPts;i++) {
    printf("xpos         = %.2f\n", h_data[i].xpos);
    printf("ypos         = %.2f\n", h_data[i].ypos);
    printf("scale        = %.2f\n", h_data[i].scale);
    printf("sharpness    = %.2f\n", h_data[i].sharpness);
    printf("edgeness     = %.2f\n", h_data[i].edgeness);
    printf("orientation  = %.2f\n", h_data[i].orientation);
    printf("score        = %.2f\n", h_data[i].score);
    auto siftData = (float*)&h_data[i].data;
    for (int j=0;j<8;j++) {
      if (j==0)
	printf("data = ");
      else
	printf("       ");
      for (int k=0;k<16;k++)
	if (siftData[j+8*k]<0.05)
	  printf(" .   ");
	else
	  printf("%.2f ", siftData[j+8*k]);
      printf("\n");
    }
  }
  printf("Number of available points: %d\n", data.numPts);
  printf("Number of allocated points: %d\n", data.maxPts);
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

double ScaleDown(const CudaImage &res, const CudaImage &src, cudaStream_t stream)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return 0.0;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  ScaleDownDenseShift<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDown<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
  checkMsg("ScaleDown() execution failed\n");
  return 0.0;
}

double ScaleUp(const CudaImage &res, const CudaImage &src, cudaStream_t stream)
{
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return 0.0;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W/2, SCALEUP_H/2);
  ScaleUp<<<blocks, threads, 0, stream>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  checkMsg("ScaleUp() execution failed\n");
  return 0.0;
}

double ComputeOrientations(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                           const TempMemory &tempMemory, int octave,
                           cudaStream_t stream)
{
  dim3 blocks(512);
#ifdef MANAGEDMEM
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.m_data, octave);
#else
#if 1
  dim3 threads(11*11);
  ComputeOrientationsCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.d_data, tempMemory.pointCounter(), octave);
#else
  dim3 threads(256);
  ComputeOrientationsCONSTNew<<<blocks, threads, 0, stream>>>(src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return 0.0;
}

double ExtractSiftDescriptors(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                              const TempMemory &tempMemory,
                              const DeviceDescriptorNormalizerData &d_normalizer,
                              float subsampling, int octave, cudaStream_t stream)
{
  dim3 blocks(512);
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsCONST<<<blocks, threads, 0, stream>>>(
      texObj, siftData.m_data, normalizer_d, subsampling, octave);
#else
  ExtractSiftDescriptorsCONSTNew<<<blocks, threads, 0, stream>>>(
      texObj, siftData.d_data, d_normalizer.get(), tempMemory.pointCounter(), subsampling, octave);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return 0.0;
}

double OrientAndExtract(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                        const TempMemory &tempMemory,
                        float subsampling, int octave, cudaStream_t stream)
{
  dim3 blocks(256);
  dim3 threads(128);
#ifdef MANAGEDMEM
  OrientAndExtractCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.m_data, subsampling, octave);
#else
  OrientAndExtractCONST<<<blocks, threads, 0, stream>>>(texObj, siftData.d_data, tempMemory.pointCounter(), subsampling, octave);
#endif
  checkMsg("OrientAndExtract() execution failed\n");
  return 0.0;
}

double RescalePositions(DeviceSiftData &siftData, float scale, cudaStream_t stream)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositions<<<blocks, threads, 0, stream>>>(siftData.d_data, siftData.numPts, scale);
  checkMsg("RescalePositions() execution failed\n");
  return 0.0;
}

double LowPass(const CudaImage &res, const CudaImage &src, cudaStream_t stream)
{
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W+2*LOWPASS_R, 4);
  LowPassBlock<<<blocks, threads, 0, stream>>>(src.d_data, res.d_data, width, pitch, height);
#else
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads, 0, stream>>>(src.d_data, res.d_data, width, pitch, height);
#endif
//  cudaStreamSynchronize(stream);
  checkMsg("LowPass() execution failed\n");
  return 0.0;
}

//==================== Multi-scale functions ===================//

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves>1) {
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=0;j<=LAPLACE_R;j++) {
      kernel[numOctaves*12*16 + 16*i + j] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += (j==0 ? 1 : 2)*kernel[numOctaves*12*16 + 16*i + j];
    }
    for (int j=0;j<=LAPLACE_R;j++)
      kernel[numOctaves*12*16 + 16*i + j] /= kernelSum;
    scale *= diffScale;
  }
}

double LaplaceMulti(const CudaImage &baseImage, const CudaImage *results,
                    int octave, cudaStream_t stream)
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W+2*LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
  LaplaceMultiMemTest<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMemOld<<<blocks, threads, 0, stream>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads, 0, stream>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
  checkMsg("LaplaceMulti() execution failed\n");
  return 0.0;
}

double FindPointsMulti(const CudaImage *sources, DeviceSiftData &siftData,
                       const TempMemory &tempMemory,
                       float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave,
                       cudaStream_t stream)
{
  if (sources->d_data==NULL) {
    printf("FindPointsMulti: missing data\n");
    return 0.0;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;
#if 0
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2, MINMAX_H);
  FindPointsMultiTest<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2);
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#else
  FindPointsMultiNew<<<blocks, threads, 0, stream>>>(sources->d_data, siftData.d_data, tempMemory.pointCounter(), w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
  checkMsg("FindPointsMulti() execution failed\n");
  return 0.0;
}

SiftData::SiftData(int num) {
  numPts = 0;
  maxPts = num;
  h_data = new SiftPoint[num];
}

DeviceSiftData::DeviceSiftData(int num) {
  numPts = 0;
  maxPts = num;
#ifdef MANAGEDMEM
  safeCall(cudaMallocManaged((void **)&m_data, sz));
#else
  d_data = nullptr;
  auto sz = sizeof(SiftPoint)*num;
  safeCall(cudaMalloc((void **)&d_data, sz));
#endif
}

SiftData::~SiftData() {
  delete[] h_data;
}

DeviceSiftData::~DeviceSiftData() {
#ifdef MANAGEDMEM
  if (m_data!=nullptr)
    safeCall(cudaFree(m_data));
#else
  if (d_data!=nullptr)
    safeCall(cudaFree(d_data));
#endif
}

SiftData &SiftData::operator=(const SiftData &other) {
  if (&other == this)
    return *this;
  if (other.maxPts > maxPts)
    throw std::invalid_argument("Target storage is smaller than destination");

  numPts = other.numPts;
  memcpy(h_data, other.h_data, numPts*sizeof(*h_data));
  return *this;
}

SiftData::SiftData(SiftData &&other) noexcept
    : numPts(other.numPts), maxPts(other.maxPts), h_data(other.h_data) {
  other.h_data = nullptr;
}

SiftData &SiftData::operator=(SiftData &&other) noexcept {
  if (&other == this)
    return *this;

  numPts = other.numPts;
  maxPts = other.maxPts;
#ifdef MANAGEDMEM
  m_data = other.m_data;
  other.m_data = nullptr;
#else
  h_data = other.h_data;
  other.h_data = nullptr;
#endif
  return *this;
}

DeviceSiftData::DeviceSiftData(DeviceSiftData &&other) noexcept
  : numPts(other.numPts), maxPts(other.numPts),
#ifdef MANAGEDMEM
    m_data(other.m_data)
#else
    d_data(other.d_data)
#endif
{
#ifdef MANAGEDMEM
  other.m_data = nullptr;
#else
  other.d_data = nullptr;
#endif
}

DeviceSiftData &DeviceSiftData::operator=(DeviceSiftData &&other) noexcept {
  if (&other == this)
    return *this;

  numPts = other.numPts;
  maxPts = other.maxPts;
#ifdef MANAGEDMEM
  m_data = other.m_data;
  other.m_data = nullptr;
#else
  d_data = other.d_data;
  other.d_data = nullptr;
#endif
  return *this;
}

void DeviceSiftData::uploadFeatures(const SiftData &src, cudaStream_t stream) {
  safeCall(cudaMemcpyAsync(d_data, src.h_data,
                           sizeof(SiftPoint) * src.numPts,
                           cudaMemcpyHostToDevice, stream));
  numPts = src.numPts;
}

void DeviceSiftData::downloadFeatures(SiftData &dst, cudaStream_t stream) const {
  safeCall(cudaMemcpyAsync(dst.h_data, d_data,
                           sizeof(SiftPoint) * numPts,
                           cudaMemcpyDeviceToHost, stream));
  dst.numPts = numPts;
}

void DeviceSiftData::uploadMatches(const SiftData &src, cudaStream_t stream) {
  float *h_ptr = &src.h_data[0].score;
  float *d_ptr = &d_data[0].score;
  safeCall(cudaMemcpy2DAsync(d_ptr, sizeof(SiftPoint), h_ptr, sizeof(SiftPoint), 5*sizeof(float), numPts, cudaMemcpyHostToDevice, stream));
  numPts = src.numPts;
}

void DeviceSiftData::downloadMatches(SiftData &dst, cudaStream_t stream) const {
  float *h_ptr = &dst.h_data[0].score;
  float *d_ptr = &d_data[0].score;
  safeCall(cudaMemcpy2DAsync(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), numPts, cudaMemcpyDeviceToHost, stream));
  dst.numPts = numPts;
}

DeviceDescriptorNormalizerData::DeviceDescriptorNormalizerData(const DescriptorNormalizerData &normalizer) {
  d_normalizer = nullptr;

  int sz = sizeof(DescriptorNormalizerData) +
           normalizer.n_steps * sizeof(int) +
           normalizer.n_data * sizeof(float);
  DescriptorNormalizerData normalizer_d;
  normalizer_d.n_steps = normalizer.n_steps;
  normalizer_d.n_data = normalizer.n_data;
  safeCall(cudaMalloc((void **)&d_normalizer, sz));
  normalizer_d.normalizer_steps = (int *)(void *)(d_normalizer + 1);
  normalizer_d.data =
      ((float *)((int *)(void *)(d_normalizer + 1) + normalizer_d.n_steps));
  cudaMemcpy(d_normalizer, &normalizer_d, sizeof(DescriptorNormalizerData),
             cudaMemcpyHostToDevice);
  cudaMemcpy(normalizer_d.normalizer_steps, normalizer.normalizer_steps,
             sizeof(int) * normalizer.n_steps, cudaMemcpyHostToDevice);
  cudaMemcpy(normalizer_d.data, normalizer.data,
             sizeof(float) * normalizer.n_data, cudaMemcpyHostToDevice);
  checkMsg("Normalizer allocation failed\n");
}

DeviceDescriptorNormalizerData::~DeviceDescriptorNormalizerData() {
  if (d_normalizer)
    safeCall(cudaFree(d_normalizer));
}

DeviceDescriptorNormalizerData::DeviceDescriptorNormalizerData(DeviceDescriptorNormalizerData &&other) noexcept
  : d_normalizer(other.d_normalizer) {
  other.d_normalizer = nullptr;
}

DeviceDescriptorNormalizerData &DeviceDescriptorNormalizerData::operator=(DeviceDescriptorNormalizerData &&other) noexcept {
  if (&other == this)
    return *this;
  d_normalizer = other.d_normalizer;
  other.d_normalizer = nullptr;
  return *this;
}
