#ifndef CAFFE_DICE_LOSS_LAYER_HPP_
#define CAFFE_DICE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
	template <typename Dtype>
	class DiceLossLayer : public LossLayer<Dtype> {
	public:
		explicit DiceLossLayer(const LayerParameter& param) : LossLayer<Dtype>(param) {}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DiceLoss"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		Blob<Dtype> predictionSum;
		Blob<Dtype> labelSum;
		Blob<Dtype> intersectionSum;

	private:
		void gpuSegmentSum(const int count, const int segmentCount, const Dtype* data, Dtype* sum);
	};
} // namespace caffe

#endif  // CAFFE_DICE_LOSS_LAYER_HPP_