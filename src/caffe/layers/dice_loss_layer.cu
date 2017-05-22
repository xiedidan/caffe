#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void ArgMax(const int n, const Dtype* data, Dtype* prediction) {
		CUDA_KERNEL_LOOP(i, n) {
			prediction[i] = data[i] > data[i + n] ? 0 : 1;
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* data = bottom[0]->gpu_data();
		const Dtype* label = bottom[1]->gpu_data();

		const int labelCount = bottom[1]->count();
		const int batchSize = bottom[1]->shape(0);
		const int dimSize = labelCount / batchSize;
		// printf("labelCount: %d, batchSize: %d, dimSize: %d, num(): %d\n", labelCount, batchSize, dimSize, bottom[1]->shape(0));

		// call cuda method to compute prediction

		for (int i = 0; i < batchSize; i++) {
			// NOLINT_NEXT_LINE(whitespace/operators)
			ArgMax<Dtype> << <CAFFE_GET_BLOCKS(dimSize), CAFFE_CUDA_NUM_THREADS >> >(
				dimSize,
				data + i * 2 * dimSize,
				bottom[1]->mutable_gpu_diff() + i * dimSize
				);
		}	
		const Dtype* prediction = bottom[1]->gpu_diff();

		caffe_gpu_set(batchSize, Dtype(0), predictionSum.mutable_gpu_data());
		caffe_gpu_set(batchSize, Dtype(0), labelSum.mutable_gpu_data());
		for (int i = 0; i < batchSize; i++) {
			caffe_gpu_asum(dimSize, prediction + i * dimSize, predictionSum.mutable_cpu_data() + i);
			caffe_gpu_asum(dimSize, label + i * dimSize, labelSum.mutable_cpu_data() + i);
		}
		
		caffe_gpu_mul(labelCount, prediction, label, bottom[1]->mutable_gpu_diff());

		caffe_gpu_set(batchSize, Dtype(0), intersectionSum.mutable_gpu_data());
		for (int i = 0; i < batchSize; i++) {
			caffe_gpu_asum(dimSize, bottom[1]->gpu_diff() + i * dimSize, intersectionSum.mutable_cpu_data() + i);
		}

		// total dice - it's simple so we directly compute on cpu
		top[0]->mutable_cpu_data()[0] = Dtype(0);
		for (int i = 0; i < batchSize; i++) {
			// printf("i: %f, p: %f, l: %f\n", intersectionSum.cpu_data()[i], predictionSum.cpu_data()[i], labelSum.cpu_data()[i]);
			top[0]->mutable_cpu_data()[0] += 2.0 * intersectionSum.cpu_data()[i] / (predictionSum.cpu_data()[i] + labelSum.cpu_data()[i]);
		}
	}

	template <typename Dtype>
	__global__ void SegmentDiff(const int dimSize, const int batchIndex, const Dtype* data, const Dtype* label, const Dtype* u, const Dtype* u2, const Dtype* intersectionSum, Dtype* diff) {
		CUDA_KERNEL_LOOP(i, dimSize) {
			// printf("data[i]: %f, data[i+labelCount]: %f\n", data[i], data[i + labelCount]);

			// diff[i] = 2.0 * ((label[i] * u) / (u * u) - 2.0 * (data[i + dimSize] * intersectionSum[batchIndex]) / (u * u));
			// diff[i + dimSize] = -2.0 * ((label[i] * u) / (u * u) - 2.0 * (data[i + dimSize] * intersectionSum[batchIndex]) / (u * u));
			diff[i] = 2.0 * ((label[i] * u[batchIndex] - 2.0 * data[i + dimSize] * intersectionSum[batchIndex]) / u2[batchIndex]);
			diff[i + dimSize] = -1.0 * diff[i];
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		if (propagate_down[0]) {
			const Dtype* data = bottom[0]->gpu_data();
			const Dtype* label = bottom[1]->gpu_data();

			const int labelCount = bottom[1]->count();
			const int batchSize = bottom[0]->shape(0);
			const int dimSize = labelCount / batchSize;

			// use label / prediction sum to save union and union ^ 2
			caffe_gpu_add(batchSize, predictionSum.gpu_data(), labelSum.gpu_data(), predictionSum.mutable_gpu_data());
			caffe_gpu_mul(batchSize, predictionSum.gpu_data(), predictionSum.gpu_data(), labelSum.mutable_gpu_data());

			for (int i = 0; i < batchSize; i++) {
				// NOLINT_NEXT_LINE(whitespace/operators)
				SegmentDiff<Dtype> << <CAFFE_GET_BLOCKS(dimSize), CAFFE_CUDA_NUM_THREADS >> >(
					dimSize, i,
					data + i * 2 * dimSize, label + i * dimSize,
					predictionSum.gpu_data(), labelSum.gpu_data(), intersectionSum.gpu_data(),
					bottom[0]->mutable_gpu_diff() + i * 2 * dimSize
					);
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DiceLossLayer);

} // namespace caffe