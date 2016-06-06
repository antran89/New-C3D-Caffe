/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef CAFFE_VOLUME_DATA_LAYER_HPP_
#define CAFFE_VOLUME_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/volume_data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class VolumeDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit VolumeDataLayer(const LayerParameter& param);
  virtual ~VolumeDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // VolumeDataLayer uses VolumeDataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "VolumeData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
  VolumeDataReader reader_;

  // develop our own DataTransformer for VolumeDataLayer, because current Caffe is
  // not supported well for other types of DataTransformer.
  Blob<Dtype> data_mean_;
  shared_ptr<Caffe::RNG> rng_;

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  /**
  * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
  *
  * @param n
  *    The upperbound (exclusive) value of the random number.
  * @return
  *    A uniformly random integer value from ({0, 1, ..., n-1}).
  */
 virtual int Rand(int n);

  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    VolumeDatum containing the data to be transformed.
   */
  vector<int> InferBlobShape(const VolumeDatum& datum);

  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data. Transform a VolumeDatum into
   * a blob of data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   * @param transformed_blob
   *    This is destination blob. It can be part of top blob's data if
   *    set_cpu_data() is used. See data_layer.cpp for an example.
   */
  void Transform(const VolumeDatum& datum, Blob<Dtype>* transformed_blob);

  // real transformation happened here
  void Transform(const VolumeDatum& datum, Dtype* transformed_data);

  // a backup of Transform function.
  // DEPRECATED: Please use the original Transform function instead
  void Transform2(const VolumeDatum& datum, Dtype* transformed_data);

};

}  // namespace caffe

#endif // CAFFE_VOLUME_DATA_LAYER_HPP

