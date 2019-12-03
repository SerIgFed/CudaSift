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
  explicit SiftData(const struct DescriptorNormalizerData &normalizer,
                    int num = 1024, bool host = false, bool dev = true,
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
  struct DescriptorNormalizerData *d_normalizer;
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

struct DescriptorNormalizerData {
  /*
   * Possible normalizer steps:
   *  0. forward internal buffer to output
   *  1. compute l2 norm and cache it
   *  2. compute l1 norm and cache it
   *  3. divide by cached norm element-wise
   *  4. clamp with alpha * accumulated_norm (consumes a single scalar alpha)
   *  5. add 128-element vector (consumes 128 scalars)
   *  6. compute matrix-vector product with 128x128 matrix (consumes
   * 128*128=16384 scalars
   *  7. divide by square root of absolute value element-wise
   *
   *  // TODO: add special handling for target cases (i.e. take positveness of
   * HoG entries into account)
   *
   *  Vanilla SIFT: 1, 4 (0.2), 1, 3, 0
   *  Vanilla RSIFT: 1, 4 (0.2), 2, 3, 0
   *  ZCA-RSIFT 1, 4 (0.2), 2, 3, 5 (-mean), 6 (ZCA), 1, 3, 0
   *  +RSIFT 1, 4 (0.2), 2, 3, 5 (-mean), 6 (ZCA), 2, 3, 7, 0
   */
  int n_steps;
  int n_data;
  int *normalizer_steps;
  float *data;
};

void InitCuda(int devNum = 0);

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves,
                 double initBlur, float thresh,
                 float lowestScale, bool scaleUp,
                 TempMemory &tempMemory);

inline
void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves,
                 double initBlur, float thresh,
                 float lowestScale = 0.0f, bool scaleUp = false) {
  TempMemory tmp(img.width, img.height, numOctaves, scaleUp);
  ExtractSift(siftData, img, numOctaves, initBlur, thresh,
              lowestScale, scaleUp, tmp);
}

void PrintSiftData(SiftData &data);
double MatchSiftData(SiftData &data1, SiftData &data2);
double FindHomography(SiftData &data,  float *homography, int *numMatches, int numLoops = 1000, float minScore = 0.85f, float maxAmbiguity = 0.95f, float thresh = 5.0f);

#endif
