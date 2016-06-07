/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */


#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/volume_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/rng.hpp"


namespace caffe {

template <typename Dtype>
VolumeDataLayer<Dtype>::VolumeDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
VolumeDataLayer<Dtype>::~VolumeDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void VolumeDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.volume_data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  VolumeDatum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  //vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_shape = InferBlobShape(datum);

  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top_shape[0] << ","
      << top_shape[1] << "," << top_shape[2] << "," << top_shape[3] << ","
      << top_shape[4];
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }

  // initialize mean value blob for our own Data Transformaner
  // check if we want to have mean
  if (this->layer_param_.volume_data_param().has_mean_file()) {
      CHECK(this->layer_param_.volume_data_param().has_mean_value() == false) <<
            "Cannot specify mean_file and mean_value at the same time";
      const string& mean_file = this->layer_param_.volume_data_param().mean_file();
      LOG(INFO) << "Loading mean file from " << mean_file;
      BlobProto blob_proto;
      ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
      data_mean_.FromProto(blob_proto);

      vector<int> shape = data_mean_.shape();
      CHECK_EQ(shape[0], 1);
      CHECK_EQ(shape[1], datum.channels());
      CHECK_EQ(shape[2], datum.length());
      CHECK_EQ(shape[3], datum.height());
      CHECK_EQ(shape[4], datum.width());
  } else {
      // Simply initialize an all-empty mean.
      vector<int> newshape(5);
      newshape[0] = 1;
      newshape[1] = datum.channels();
      newshape[2] = datum.length();
      newshape[3] = datum.height();
      newshape[4] = datum.width();
      data_mean_.Reshape(newshape);
      if (this->layer_param_.volume_data_param().has_mean_value()) {
          LOG(INFO) << "Using mean value of " << this->layer_param_.volume_data_param().mean_value();
          caffe::caffe_set(data_mean_.count(), (Dtype)this->layer_param_.volume_data_param().mean_value(),
                           (Dtype*)data_mean_.mutable_cpu_data());
      }
  }

  // Initialize random number generator
  this->InitRand();
}

// This function is called on prefetch thread
template<typename Dtype>
void VolumeDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.volume_data_param().batch_size();
  VolumeDatum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  //vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> top_shape = InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    VolumeDatum& datum = *(reader_.full().pop("Waiting for data"));
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    vector<int> indices(1);     // vector index is more general in VolumeDatum
    indices[0] = item_id;
    int offset = batch->data_.offset(indices);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();

    reader_.free().push(const_cast<VolumeDatum*>(&datum));
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

/**
 * @brief Infers the shape of transformed_blob will have when
 *    the transformation is applied to the data.
 *
 * @param datum
 *    VolumeDatum containing the data to be transformed.
 */
template<typename Dtype>
vector<int> VolumeDataLayer<Dtype>::InferBlobShape(const VolumeDatum& datum)
{
    const int crop_size = this->layer_param_.volume_data_param().crop_size();
    const int datum_channels = datum.channels();
    const int datum_length = datum.length();
    const int datum_height = datum.height();
    const int datum_width = datum.width();
    // Check dimensions
    CHECK_GT(datum_channels, 0);
    CHECK_GE(datum_height, crop_size);
    CHECK_GE(datum_width, crop_size);
    // Build BlobShape
    std::vector<int> shape(5);
    shape[0] = 1;
    shape[1] = datum_channels;
    shape[2] = datum_length;
    shape[3] = (crop_size) ? crop_size : datum_height;
    shape[4] = (crop_size) ? crop_size : datum_width;
    return shape;
}

template <typename Dtype>
void VolumeDataLayer<Dtype>::InitRand() {
  const bool needs_rand = this->layer_param_.volume_data_param().mirror() ||
      (this->phase_ == TRAIN && this->layer_param_.volume_data_param().crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int VolumeDataLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

/**
 * @brief Applies the transformation defined in the data layer's
 * transform_param block to the data.
 *
 * @param datum
 *    VolumeDatum containing the data to be transformed.
 * @param transformed_blob
 *    This is destination blob. It can be part of top blob's data if
 *    set_cpu_data() is used. See data_layer.cpp for an example.
 */
template<typename Dtype>
void VolumeDataLayer<Dtype>::Transform(const VolumeDatum& datum, Blob<Dtype>* transformed_blob)
{
    const int crop_size = this->layer_param_.volume_data_param().crop_size();
    const int datum_channels = datum.channels();
    const int datum_length = datum.length();
    const int datum_height = datum.height();
    const int datum_width = datum.width();

    // Check dimensions.
    // shape [N*C*L*H*W]
    vector<int> blob_shape = transformed_blob->shape();

    CHECK_GE(blob_shape[0], 1);
    CHECK_EQ(blob_shape[1], datum_channels);
    CHECK_EQ(blob_shape[2], datum_length);
    CHECK_LE(blob_shape[3], datum_height);
    CHECK_LE(blob_shape[4], datum_width);

    if (crop_size) {
      CHECK_EQ(crop_size, blob_shape[3]);
      CHECK_EQ(crop_size, blob_shape[4]);
    } else {
      CHECK_EQ(datum_height, blob_shape[3]);
      CHECK_EQ(datum_width, blob_shape[4]);
    }

    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    Transform(datum, transformed_data);
}

// a shorter, concrete implementation of Transform function.
template<typename Dtype>
void VolumeDataLayer<Dtype>::Transform(const VolumeDatum& datum, Dtype* transformed_data)
{
    // some necessary variables
    const string& data = datum.data();
    const Dtype* mean = this->data_mean_.cpu_data();
    const int crop_size = this->layer_param_.volume_data_param().crop_size();
    const int datum_height = datum.height();
    const int datum_width = datum.width();
    const bool mirror = this->layer_param_.volume_data_param().mirror();
    const bool do_mirror = mirror && Rand(2);
    const int channels = datum.channels();
    const int length = datum.length();
    const Dtype scale = this->layer_param_.volume_data_param().scale();
    const bool has_uint8 = data.size() > 0;
    const bool has_mean = this->layer_param_.volume_data_param().has_mean_file() ||
            this->layer_param_.volume_data_param().has_mean_value();

    // main transformation: transform
    int height = datum_height;      // height, width of a final patch
    int width = datum_width;

    int h_off = 0, w_off = 0;

    if (crop_size) {
        height = crop_size;
        width = crop_size;
        // We only do random crop when we do training.
        if (this->phase_ == caffe::TRAIN) {
            h_off = Rand(datum_height - crop_size + 1);
            w_off = Rand(datum_width - crop_size + 1);
        } else {
            h_off = (datum_height - crop_size) / 2;
            w_off = (datum_width - crop_size) / 2;
        }
    }

    // mirroring, cropping, scaling
    Dtype datum_element;
    int top_index, data_index;
    for (int c = 0; c < channels; ++c) {
        for (int l = 0; l < length; ++l) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    data_index = ((c * length + l) * datum_height + h + h_off) * datum_width + w + w_off;
                    if (do_mirror)
                        top_index = ((c * length + l) * height + h) * width + (width - 1 - w);
                    else
                        top_index = ((c * length + l) * height + h) * width + w;
                    if (has_uint8)
                        datum_element = static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                    else
                        datum_element = datum.float_data(data_index);
                    if (has_mean)
                        transformed_data[top_index] =
                          (datum_element - mean[data_index]) * scale;
                    else
                        transformed_data[top_index] = datum_element * scale;
                }
            }
        }
    }
}

template<typename Dtype>
void VolumeDataLayer<Dtype>::Transform2(const VolumeDatum& datum, Dtype* transformed_data)
{
    // some necessary variables
    const Dtype* mean = this->data_mean_.cpu_data();
    const int crop_size = this->layer_param_.volume_data_param().crop_size();
    const int height = datum.height();
    const int width = datum.width();
    const bool mirror = this->layer_param_.volume_data_param().mirror();
    const bool do_mirror = mirror && Rand(2);
    const int channels = datum.channels();
    const int length = datum.length();
    const Dtype scale = this->layer_param_.volume_data_param().scale();
    const int show_data = this->layer_param_.volume_data_param().show_data();
    const int size = channels * length * height * width;

    char *data_buffer;
    if (show_data)
        data_buffer = new char[size];

    // main transformation: transform
    const string& data = datum.data();
    if (crop_size) {
        CHECK(data.size()) << "Image cropping only support uint8 data";
        int h_off, w_off;
        // We only do random crop when we do training.
        if (this->phase_ == caffe::TRAIN) {
            h_off = Rand(height - crop_size + 1);
            w_off = Rand(width - crop_size + 1);
        } else {
            h_off = (height - crop_size) / 2;
            w_off = (width - crop_size) / 2;
        }
        if (do_mirror) {
            // Copy mirrored version
            for (int c = 0; c < channels; ++c) {
                for (int l = 0; l < length; ++l) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int top_index = ((c * length + l) * crop_size + h)
                                    * crop_size + (crop_size - 1 - w);
                            int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                            Dtype datum_element =
                                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                            transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
                            if (show_data)
                                data_buffer[((c * length + l) * crop_size + h)
                                        * crop_size + (crop_size - 1 - w)] = static_cast<uint8_t>(data[data_index]);
                        }
                    }
                }
            }
        } else {
            // Normal copy
            for (int c = 0; c < channels; ++c) {
                for (int l = 0; l < length; ++l) {
                    for (int h = 0; h < crop_size; ++h) {
                        for (int w = 0; w < crop_size; ++w) {
                            int top_index = ((c * length + l) * crop_size + h)
                                    * crop_size + w;
                            int data_index = ((c * length + l) * height + h + h_off) * width + w + w_off;
                            Dtype datum_element =
                                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                            transformed_data[top_index] = (datum_element - mean[data_index]) * scale;
                            if (show_data)
                                data_buffer[((c * length + l) * crop_size + h)
                                        * crop_size + w] = static_cast<uint8_t>(data[data_index]);
                        }
                    }
                }
            }
        }
    } else {
        // we will prefer to use data() first, and then try float_data()
        if (data.size()) {
            for (int j = 0; j < size; ++j) {
                Dtype datum_element =
                        static_cast<Dtype>(static_cast<uint8_t>(data[j]));
                transformed_data[j] = (datum_element - mean[j]) * scale;
                if (show_data)
                    data_buffer[j] = static_cast<uint8_t>(data[j]);
            }
        } else {
            for (int j = 0; j < size; ++j) {
                transformed_data[j] =
                        (datum.float_data(j) - mean[j]) * scale;
            }
        }
    }

    if (show_data>0) {
        int image_size, channel_size;
        if (crop_size){
            image_size = crop_size * crop_size;
        }else{
            image_size = height * width;
        }
        channel_size = length * image_size;
        for (int c = 0; c < channels; ++c) {
            for (int l = 0; l < length; ++l) {
                cv::Mat img;
                char ch_name[64];
                if (crop_size)
                    BufferToGrayImage(data_buffer + c * channel_size + l * image_size, crop_size, crop_size, &img);
                else
                    BufferToGrayImage(data_buffer + c * channel_size + l * image_size, height, width, &img);
                sprintf(ch_name, "Channel %d", c);
                cv::imshow(ch_name, img);
            }
            cv::waitKey(100);
        }
    }

}

INSTANTIATE_CLASS(VolumeDataLayer);
REGISTER_LAYER_CLASS(VolumeData);

}  // namespace caffe
