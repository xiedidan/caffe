#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	__global__ void ArgMax(const int n, const Dtype* score, Dtype* predictions) {
		CUDA_KERNEL_LOOP(i, n) {
			predictions[i] = score[i] >= score[i + n] ? 0 : 1;
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* data = bottom[0]->gpu_data();
		const Dtype* label = bottom[1]->gpu_data();

		const int labelCount = bottom[1]->count();
		const int batchSize = bottom[1]->num();
		const int dimSize = labelCount / batchSize;

		// use bottom[1]->gpu_diff to save gpu memory
		Dtype* prediction = bottom[1]->mutable_gpu_diff();

		// call cuda method to compute prediction
		// NOLINT_NEXT_LINE(whitespace/operators)
		ArgMax<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			labelCount,
			data,
			prediction
		);

		predictionSum.clear();
		labelSum.clear();
		for (int i = 0; i < batchSize; i++) {
			Dtype currPredictionSum;
			Dtype currLabelSum;

			caffe_gpu_asum(dimSize, prediction + i * dimSize, &currPredictionSum);
			caffe_gpu_asum(dimSize, label + i * dimSize, &currPredictionSum);
			
			predictionSum.push_back(currPredictionSum);
			labelSum.push_back(currLabelSum);
		}
		
		Dtype* intersection = bottom[1]->mutable_gpu_diff();
		caffe_gpu_mul(labelCount, prediction, label, intersection);

		intersectionSum.clear();
		for (int i = 0; i < batchSize; i++) {
			Dtype currIntersectionSum;
			caffe_gpu_asum(dimSize, intersection + i * dimSize, &currIntersectionSum);
			intersectionSum.push_back(currIntersectionSum);

			// total dice - it's simple so we directly compute on cpu
			top[0]->mutable_cpu_data()[0] = 2.0 * currIntersectionSum / (predictionSum[i] + labelSum[i]);
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
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

			for (int i = 0; i < batchSize; i++) {
				Dtype currUnion = predictionSum[i] + labelSum[i];
				Dtype currIntersection = intersectionSum[i];

				for (int j = 0; j < dimSize; j++) {
					// we always have 2 channels for dice
					Dtype currLabel = label[i * dimSize + j];
					Dtype currData1 = data[(i * 2) * dimSize + j];
					Dtype currData2 = data[(i * 2 + 1) * dimSize + j];

					bottom[0]->mutable_cpu_diff()[(i * 2) * dimSize + j] =
						2.0 * ((currLabel * currUnion) / (currUnion * currUnion) - 2.0 * currData1 * currIntersection / (currUnion * currUnion));
					bottom[0]->mutable_cpu_diff()[(i * 2 + 1) * dimSize + j] =
						-2.0 * ((currLabel * currUnion) / (currUnion * currUnion) - 2.0 * currData1 * currIntersection / (currUnion * currUnion));
				}
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(DiceLossLayer);
}