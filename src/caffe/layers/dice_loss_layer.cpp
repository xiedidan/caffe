#include <vector>

#include "caffe/layers/dice_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void DiceLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		vector<int> topShape(0);
		top[0]->Reshape(topShape);

		vector<int> sumShape(1);
		sumShape[0] = bottom[1]->num();
		predictionSum.Reshape(sumShape);
		labelSum.Reshape(sumShape);
		intersectionSum.Reshape(sumShape);
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
		const Dtype* prediction = bottom[1]->cpu_diff();

		for (int i = 0; i < batchSize; i++) {
			predictionSum.mutable_cpu_data()[i] = caffe_cpu_asum(dimSize, prediction + i * dimSize);
			labelSum.mutable_cpu_data()[i] = caffe_cpu_asum(dimSize, label + i * dimSize);
		}

		// again, use bottom[1]->cpu_diff to save memory
		Dtype* intersection = bottom[1]->mutable_cpu_diff();
		caffe_mul(labelCount, prediction, label, intersection);

		top[0]->mutable_cpu_data()[0] = Dtype(0);
		for (int i = 0; i < batchSize; i++) {
			intersectionSum.mutable_cpu_data()[i] = caffe_cpu_asum(dimSize, intersection + i * dimSize);

			// total dice
			top[0]->mutable_cpu_data()[0] += 2.0 * intersectionSum.cpu_data()[i] / (predictionSum.cpu_data()[i] + labelSum.cpu_data()[i]);
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
				Dtype currUnion = predictionSum.cpu_data()[i] + labelSum.cpu_data()[i];
				Dtype currIntersection = intersectionSum.cpu_data()[i];
				for (int j = 0; j < dimSize; j++) {
					// we always have 2 channels for dice
					Dtype currLabel = label[i * dimSize + j];
					Dtype currData1 = data[i * dimSize + j];
					Dtype currData2 = data[i * dimSize + j + labelCount];

					// printf("currData1: %f, currData2: %f\n", currData1, currData2);

					bottom[0]->mutable_cpu_diff()[i * dimSize + j] = 
						2.0 * ((currLabel * currUnion) / (currUnion * currUnion) - 2.0 * currData2 * currIntersection / (currUnion * currUnion));
					bottom[0]->mutable_cpu_diff()[i * dimSize + j + labelCount] = 
						-2.0 * ((currLabel * currUnion) / (currUnion * currUnion) - 2.0 * currData2 * currIntersection / (currUnion * currUnion));
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

