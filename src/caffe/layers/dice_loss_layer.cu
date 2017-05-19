#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void ArgMax(const int n, const Dtype* data, Dtype* prediction) {
		CUDA_KERNEL_LOOP(i, n) {
			prediction[i] = data[i] >= data[i + n] ? 1 : 0;
		}
	}

	template <typename Dtype>
	__global__ void SegmentSum(const int count, const int segmentSize, const Dtype* data, Dtype* sum) {
		CUDA_KERNEL_LOOP(i, count) {
			// TODO : faster implementation?
			int sumIndex = i / segmentSize;
			sum[sumIndex] += data[i];
		}
	}

	template <typename Dtype>
	__global__ void gpuMemset(const int count, Dtype* data, const Dtype value) {
		CUDA_KERNEL_LOOP(i, count) {
			data[i] = value;
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* data = bottom[0]->gpu_data();
		const Dtype* label = bottom[1]->gpu_data();

		const int labelCount = bottom[1]->count();
		const int batchSize = bottom[1]->num();
		const int dimSize = labelCount / batchSize;

		// call cuda method to compute prediction

		// NOLINT_NEXT_LINE(whitespace/operators)
		ArgMax<Dtype> <<<CAFFE_GET_BLOCKS(labelCount), CAFFE_CUDA_NUM_THREADS>>>(
			labelCount,
			data,
			bottom[1]->mutable_gpu_diff()
		);
		const Dtype* prediction = bottom[1]->gpu_diff();
		
		caffe_gpu_asum(labelCount, prediction, predictionSum.mutable_cpu_data());
		caffe_gpu_asum(labelCount, label, labelSum.mutable_cpu_data());
		caffe_gpu_mul(labelCount, prediction, label, bottom[1]->mutable_gpu_diff());
		caffe_gpu_asum(labelCount, bottom[1]->gpu_diff(), intersectionSum.mutable_cpu_data());
		top[0]->mutable_cpu_data()[0] = 2.0 * intersectionSum.cpu_data()[0] / (predictionSum.cpu_data()[0] + labelSum.cpu_data()[0]);
		
		/*
		// NOLINT_NEXT_LINE(whitespace/operators)
		gpuMemset<Dtype> << <CAFFE_GET_BLOCKS(batchSize), CAFFE_CUDA_NUM_THREADS >> >(
			batchSize,
			predictionSum.mutable_gpu_data(),
			Dtype(0)
		);

		// NOLINT_NEXT_LINE(whitespace/operators)
		SegmentSum<Dtype> <<<CAFFE_GET_BLOCKS(labelCount), CAFFE_CUDA_NUM_THREADS>>>(
			labelCount,
			dimSize,
			prediction,
			predictionSum.mutable_gpu_data()
		);

		// NOLINT_NEXT_LINE(whitespace/operators)
		gpuMemset<Dtype> << <CAFFE_GET_BLOCKS(batchSize), CAFFE_CUDA_NUM_THREADS >> >(
			batchSize,
			labelSum.mutable_gpu_data(),
			Dtype(0)
		);

		// NOLINT_NEXT_LINE(whitespace/operators)
		SegmentSum<Dtype> <<<CAFFE_GET_BLOCKS(labelCount), CAFFE_CUDA_NUM_THREADS>>>(
			labelCount,
			dimSize,
			label,
			labelSum.mutable_gpu_data()
		);
		
		caffe_gpu_mul(labelCount, prediction, label, bottom[1]->mutable_gpu_diff());

		// NOLINT_NEXT_LINE(whitespace/operators)
		gpuMemset<Dtype> << <CAFFE_GET_BLOCKS(batchSize), CAFFE_CUDA_NUM_THREADS >> >(
			batchSize,
			intersectionSum.mutable_gpu_data(),
			Dtype(0)
		);

		// NOLINT_NEXT_LINE(whitespace/operators)
		SegmentSum<Dtype> <<<CAFFE_GET_BLOCKS(labelCount), CAFFE_CUDA_NUM_THREADS>>>(
			labelCount,
			dimSize,
			bottom[1]->gpu_diff(),
			intersectionSum.mutable_gpu_data()
		);

		// total dice - it's simple so we directly compute on cpu
		top[0]->mutable_cpu_data()[0] = Dtype(0);
		for (int i = 0; i < batchSize; i++) {
			 // printf("i: %f, p: %f, l: %f\n", intersectionSum.cpu_data()[i], predictionSum.cpu_data()[i], labelSum.cpu_data()[i]);
			top[0]->mutable_cpu_data()[0] += 2.0 * intersectionSum.cpu_data()[i] / (predictionSum.cpu_data()[i] + labelSum.cpu_data()[i]);
		}
		*/
	}

	template <typename Dtype>
	__global__ void SegmentDiff(const int count, const Dtype* data, const Dtype* label, const Dtype* predictionSum, const Dtype* labelSum, const Dtype* intersectionSum, Dtype* diff) {
		CUDA_KERNEL_LOOP(i, count) {
			Dtype u = predictionSum[0] + labelSum[0];

			diff[i] = 2.0 * ((label[i] * u) / (u * u) - 2.0 * (data[i] * intersectionSum[0]) / (u * u));
			diff[i + count] = -2.0 * ((label[i] * u) / (u * u) - 2.0 * (data[i + count] * intersectionSum[0]) / (u * u));
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
			const int batchSize = bottom[0]->num();
			const int dimSize = labelCount / batchSize;

			// NOLINT_NEXT_LINE(whitespace/operators)
			SegmentDiff<Dtype> <<<CAFFE_GET_BLOCKS(labelCount), CAFFE_CUDA_NUM_THREADS>>>(
				labelCount, 
				data, label,
				predictionSum.gpu_data(), labelSum.gpu_data(), intersectionSum.gpu_data(),
				bottom[0]->mutable_gpu_diff()
			);
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DiceLossLayer);

} // namespace caffe