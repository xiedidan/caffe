#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void DiceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> topShape(0);
		top[0]->Reshape(topShape);
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* data = bottom[0]->cpu_data();
		const Dtype* label = bottom[1]->cpu_data();

		const int labelCount = bottom[1]->count();
		const int batchSize = bottom[1]->num();
		const int dimSize = labelCount / batchSize;

		// bottom[1]->cpu_diff is not used (for backward), so use it to save memory
		for (int i = 0; i < labelCount; i++) {
			bottom[1]->mutable_cpu_diff()[i] = data[i] >= data[i + labelCount] ? 0 : 1;
		}
		Dtype* prediction = bottom[1]->cpu_diff();

		top[0]->mutable_cpu_data()[0] = Dtype(0);
		predictionSum.clear();
		labelSum.clear();
		for (int i = 0; i < batchSize; i++) {
			Dtype currPredictionSum = caffe_cpu_asum(dimSize, prediction + i * dimSize);
			predictionSum.push_back(currPredictionSum);

			Dtype currLabelSum = caffe_cpu_asum(dimSize, label + i * dimSize);
			labelSum.push_back(currLabelSum);
		}

		// again, use bottom[1]->cpu_diff to save memory
		Dtype* intersection = bottom[1]->mutable_cpu_diff();
		caffe_mul(labelCount, prediction, label, intersection);

		intersectionSum.clear();
		top[0]->mutable_cpu_data()[0] = Dtype(0);
		for (int i = 0; i < batchSize; i++) {
			Dtype currIntersectionSum = caffe_cpu_asum(dimSize, intersection + i * dimSize);
			intersectionSum.push_back(currIntersectionSum);

			// total dice
			top[0]->mutable_cpu_data()[0] += 2.0 * currIntersectionSum / (predictionSum[i] + labelSum[i]);
		}
	}

	template <typename Dtype>
	void DiceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type()
				<< " Layer cannot backpropagate to label inputs.";
		}

		if (propagate_down[0]) {
			const Dtype* data = bottom[0]->cpu_data();
			const Dtype* label = bottom[1]->cpu_data();

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

#ifdef CPU_ONLY
	STUB_GPU(DiceLossLayer);
#endif
	
	INSTANTIATE_CLASS(DiceLossLayer);
	REGISTER_LAYER_CLASS(DiceLoss);

} // namespace caffe

