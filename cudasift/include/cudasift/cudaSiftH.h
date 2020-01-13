#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudasift/cudautils.h"
#include "cudasift/cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

int ExtractSiftLoop(SiftData &siftData, const CudaImage &img,
                    const DeviceDescriptorNormalizerData &d_normalizer,
                    int numOctaves, double initBlur, float thresh, float lowestScale,
                    float subsampling, TempMemory &memorySub);
void ExtractSiftOctave(SiftData &siftData, const CudaImage &img,
                       const DeviceDescriptorNormalizerData &d_normalizer,
                       int octave, float thresh, float lowestScale,
                       float subsampling, TempMemory &memoryTmp);
double ScaleDown(const SiftData &siftData, const CudaImage &res, const CudaImage &src);
double ScaleUp(const SiftData &siftData, const CudaImage &res, const CudaImage &src);
double ComputeOrientations(cudaTextureObject_t texObj, const CudaImage &src, SiftData &siftData,
                           const TempMemory &tempMemory, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData,
                              const TempMemory &tempMemory,
                              const DeviceDescriptorNormalizerData &d_normalizer,
                              float subsampling, int octave);
double OrientAndExtract(cudaTextureObject_t texObj, SiftData &siftData,
                        const TempMemory tempMemory, float subsampling, int octave);
double RescalePositions(SiftData &siftData, float scale);
double LowPass(const SiftData &siftData, const CudaImage &res, const CudaImage &src);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(const SiftData &siftData, cudaTextureObject_t texObj, const CudaImage &baseImage, const CudaImage *results, int octave);
double FindPointsMulti(const CudaImage *sources, SiftData &siftData,
                       const TempMemory &tempMemory,
                       float thresh, float edgeLimit, float factor,
                       float lowestScale, float subsampling, int octave);

#endif
