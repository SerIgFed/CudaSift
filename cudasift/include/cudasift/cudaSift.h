#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudasift/cudaImage.h"
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
  explicit SiftData(int num = 1024);
  ~SiftData();
  SiftData(const SiftData &) = delete;
  SiftData &operator=(const SiftData &);
  SiftData(SiftData &&other) noexcept;
  SiftData &operator=(SiftData &&other) noexcept;

  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
  SiftPoint *h_data;  // Host (CPU) data
};

struct DeviceSiftData {
  explicit DeviceSiftData(int num = 1024);
  void uploadFeatures(const SiftData &src, cudaStream_t stream = 0);
  void uploadMatches(const SiftData &src, cudaStream_t stream = 0);
  void downloadFeatures(SiftData &dst, cudaStream_t stream = 0) const;
  void downloadMatches(SiftData &dst, cudaStream_t stream = 0) const;

  ~DeviceSiftData();
  DeviceSiftData(const DeviceSiftData &) = delete;
  DeviceSiftData &operator=(const DeviceSiftData &) = delete;
  DeviceSiftData(DeviceSiftData &&other) noexcept;
  DeviceSiftData &operator=(DeviceSiftData &&other) noexcept;

  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *d_data;  // Device (GPU) data
#endif
};

class TempMemory {
public:
  float *laplaceBuffer() const { return d_data; }
  CudaImage image(int octave, cudaStream_t stream = 0) const;
  cudaTextureObject_t texture(int octave) const;
  unsigned int *pointCounter() const { return d_PointCounter; }

  explicit operator bool() const { return d_data; }
  TempMemory(int width, int height, int num_octaves, bool scale_up = false);
  TempMemory(const TempMemory &other) = delete;
  TempMemory &operator =(const TempMemory &other) = delete;
  TempMemory(TempMemory &&other) noexcept;
  TempMemory &operator =(TempMemory &&other) noexcept;
  ~TempMemory();
  void setSize(int w, int h);

private:
  float *imageBuffer() const { return d_data + laplace_buffer_size; }

  std::vector<cudaTextureObject_t> textures;
  float *d_data = nullptr;
  size_t laplace_buffer_size;
  int width, height;
  int restrict_width, restrict_height;
  int num_octaves;
  unsigned int *d_PointCounter;
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

class DeviceDescriptorNormalizerData {
public:
  explicit DeviceDescriptorNormalizerData(const DescriptorNormalizerData &normalizer);
  ~DeviceDescriptorNormalizerData();
  DeviceDescriptorNormalizerData(const DeviceDescriptorNormalizerData &) = delete;
  DeviceDescriptorNormalizerData &operator=(const DeviceDescriptorNormalizerData &) = delete;
  DeviceDescriptorNormalizerData(DeviceDescriptorNormalizerData &&) noexcept;
  DeviceDescriptorNormalizerData &operator=(DeviceDescriptorNormalizerData &&) noexcept;

  const DescriptorNormalizerData *get() const { return d_normalizer; }

private:
  DescriptorNormalizerData *d_normalizer;
};

void InitCuda(int maxPts, int numOctaves, float initBlur, int devNum = 0);

void ExtractSift(DeviceSiftData &siftData,
                 const DeviceDescriptorNormalizerData &d_normalizer,
                 const CudaImage &img, int numOctaves, float thresh,
                 float lowestScale, bool scaleUp,
                 TempMemory &tempMemory, cudaStream_t stream = 0);

void ExtractSift(DeviceSiftData &siftData,
                 const DeviceDescriptorNormalizerData &d_normalizer,
                 const CudaImage &detect_img, const CudaImage &extract_img,
                 int numOctaves, float thresh, float lowestScale, bool scaleUp,
                 TempMemory &tempMemory, cudaStream_t stream = 0);

inline
void ExtractSift(DeviceSiftData &siftData,
                 const DeviceDescriptorNormalizerData &d_normalizer,
                 const CudaImage &img, int numOctaves, float thresh,
                 float lowestScale = 0.0f, bool scaleUp = false,
                 cudaStream_t stream = 0) {
  TempMemory tmp(img.width, img.height, numOctaves, scaleUp);
  ExtractSift(siftData, d_normalizer, img, numOctaves, thresh,
              lowestScale, scaleUp, tmp, stream);
}

void PrintSiftData(SiftData &data);
double MatchSiftData(const DeviceSiftData &data1, const DeviceSiftData &data2, cudaStream_t stream = 0);
double FindHomography(DeviceSiftData &data,  float *homography, int *numMatches,
                      int numLoops = 1000, float minScore = 0.85f,
                      float maxAmbiguity = 0.95f, float thresh = 5.0f,
                      cudaStream_t stream = 0);

#endif
