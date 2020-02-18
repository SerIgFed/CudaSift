#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudasift/cudautils.h"
#include "cudasift/cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

int ComputeSiftLoop(DeviceSiftData &siftData, const CudaImage &img,
                    const DeviceDescriptorNormalizerData &d_normalizer,
                    int numOctaves, double initBlur, float thresh, float lowestScale,
                    float subsampling, TempMemory &memorySub, cudaStream_t stream);
void ComputeSiftOctave(DeviceSiftData &siftData, const CudaImage &img,
                       const DeviceDescriptorNormalizerData &d_normalizer,
                       int octave, float thresh, float lowestScale,
                       float subsampling, TempMemory &memoryTmp, cudaStream_t stream);
int ExtractSiftLoop(DeviceSiftData &siftData, const CudaImage &img,
                    const DeviceDescriptorNormalizerData &d_normalizer,
                    int numOctaves, double initBlur, float subsampling,
                    TempMemory &memorySub, bool scaled, cudaStream_t stream);
void ExtractSiftOctave(DeviceSiftData &siftData, const CudaImage &img,
                       const DeviceDescriptorNormalizerData &d_normalizer,
                       int octave, float subsampling, TempMemory &memoryTmp,
                       cudaStream_t stream);
double ScaleDown(const CudaImage &res, const CudaImage &src, cudaStream_t stream);
double ScaleUp(const CudaImage &res, const CudaImage &src, cudaStream_t stream);
double ComputeOrientations(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                           const TempMemory &tempMemory, int octave, cudaStream_t stream);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                              const TempMemory &tempMemory,
                              const DeviceDescriptorNormalizerData &d_normalizer,
                              float subsampling, int octave, cudaStream_t stream);
double OrientAndExtract(cudaTextureObject_t texObj, DeviceSiftData &siftData,
                        const TempMemory tempMemory, float subsampling, int octave, cudaStream_t stream);
double RescalePositions(DeviceSiftData &siftData, float scale, cudaStream_t stream);
double LowPass(const CudaImage &res, const CudaImage &src, cudaStream_t stream);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(const CudaImage &baseImage, const CudaImage *results, int octave, cudaStream_t stream);
double FindPointsMulti(const CudaImage *sources, DeviceSiftData &siftData,
                       const TempMemory &tempMemory,
                       float thresh, float edgeLimit, float factor,
                       float lowestScale, float subsampling, int octave, cudaStream_t stream);

#endif
