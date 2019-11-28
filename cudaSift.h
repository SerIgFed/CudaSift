#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include <vector>

struct SiftPoint {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  alignas(16) float data[128];
};

struct SiftData {
  explicit SiftData(int num = 1024, bool host = false, bool dev = true,
                    cudaStream_t stream = 0);
  ~SiftData();
  SiftData(const SiftData &) = delete;
  SiftData &operator=(const SiftData &) = delete;
  SiftData(SiftData &&other) noexcept;
  SiftData &operator=(SiftData &&other) noexcept;

  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
  cudaStream_t stream;
#endif
  unsigned int *d_PointCounter;
};

class TempMemory {
public:
  float *laplaceBuffer() const { return d_data; }
  CudaImage image(int octave, cudaStream_t stream = 0) const;
  cudaTextureObject_t texture(int octave) const;

  operator bool() const { return d_data; }
  TempMemory(int width, int height, int num_octaves, bool scale_up = false);
  TempMemory(const TempMemory &other) = delete;
  TempMemory &operator =(const TempMemory &other) = delete;
  TempMemory(TempMemory &&other) noexcept;
  TempMemory &operator =(TempMemory &&other) noexcept;
  ~TempMemory();

private:
  float *imageBuffer() const { return d_data + laplace_buffer_size; }

  std::vector<cudaTextureObject_t> textures;
  float *d_data = nullptr;
  size_t laplace_buffer_size;
  int width, height;
  int num_octaves;
};

void InitCuda(int devNum = 0);
void ExtractSift(SiftData &siftData, const CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, TempMemory &tempMemory);
inline void ExtractSift(SiftData &siftData, const CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale = 0.0f, bool scaleUp = false) {
  TempMemory tmp(img.width, img.height, numOctaves, scaleUp);
  ExtractSift(siftData, img, numOctaves, initBlur, thresh, lowestScale, scaleUp, tmp);
}
void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
