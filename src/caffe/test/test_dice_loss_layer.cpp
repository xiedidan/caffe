#include <cmath>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dice_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define BATCH_SIZE 2
#define DIM_SIZE 4

namespace caffe {
	template <typename TypeParam>
	class DiceLossLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		DiceLossLayerTest() :
			blob_bottom_data_(new Blob<Dtype>(BATCH_SIZE, 2, DIM_SIZE, 1)),
			blob_bottom_label_(new Blob<Dtype>(BATCH_SIZE, 1, DIM_SIZE, 1)),
			blob_top_loss_(new Blob<Dtype>()) {
			FillerParameter filler_param;
			GaussianFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_data_);
			filler.Fill(this->blob_bottom_label_);

			blob_bottom_vec_.push_back(blob_bottom_data_);
			blob_bottom_vec_.push_back(blob_bottom_label_);
			blob_top_vec_.push_back(blob_top_loss_);
		}

		virtual ~DiceLossLayerTest() {
			delete blob_bottom_data_;
			delete blob_bottom_label_;
			delete blob_top_loss_;
		}

		void TestForward() {
			Dtype* data = blob_bottom_data_->mutable_cpu_data();
			Dtype* label = blob_bottom_label_->mutable_cpu_data();
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < DIM_SIZE / 2; j++) {
					data[i * DIM_SIZE + j * 2] = Dtype(1);
					data[i * DIM_SIZE + j * 2 + BATCH_SIZE * DIM_SIZE] = Dtype(0);
					label[i * DIM_SIZE + j * 2] = Dtype(1);
				}

				for (int j = 0; j < 2; j++) {
					data[i * DIM_SIZE + j * 2 + 1] = Dtype(0);
					data[i * DIM_SIZE + j * 2 + 1 + BATCH_SIZE * DIM_SIZE] = Dtype(1);
					label[i * DIM_SIZE + j * 2 + 1] = Dtype(1);
				}
			}

			LayerParameter layerParam;
			DiceLossLayer<Dtype> diceLossLayer(layerParam);

			diceLossLayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			diceLossLayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			const Dtype loss = blob_top_loss_->cpu_data()[0];
			EXPECT_NEAR(Dtype(2.0 * BATCH_SIZE / 3.0), loss, Dtype(0.0001));
		}

		void TestBackward() {
			Dtype* data = blob_bottom_data_->mutable_cpu_data();
			Dtype* label = blob_bottom_label_->mutable_cpu_data();
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < DIM_SIZE / 2; j++) {
					data[i * DIM_SIZE + j * 2] = Dtype(1);
					data[i * DIM_SIZE + j * 2 + BATCH_SIZE * DIM_SIZE] = Dtype(0);
					label[i * DIM_SIZE + j * 2] = Dtype(1);
				}

				for (int j = 0; j < 2; j++) {
					data[i * DIM_SIZE + j * 2 + 1] = Dtype(0);
					data[i * DIM_SIZE + j * 2 + 1 + BATCH_SIZE * DIM_SIZE] = Dtype(1);
					label[i * DIM_SIZE + j * 2 + 1] = Dtype(1);
				}
			}

			LayerParameter layerParam;
			DiceLossLayer<Dtype> diceLossLayer(layerParam);

			diceLossLayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			diceLossLayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			vector<bool> propagate_down(2);
			propagate_down[0] = true;
			propagate_down[1] = false;
			diceLossLayer.Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
			
			const Dtype* diff = this->blob_bottom_vec_[0]->cpu_diff();
			for (int i = 0; i < BATCH_SIZE; i++) {
				for (int j = 0; j < DIM_SIZE; j++) {
					EXPECT_NEAR(Dtype(1), diff[i * DIM_SIZE + j], Dtype(0.0001));
					EXPECT_NEAR(Dtype(1), diff[i * DIM_SIZE + j + DIM_SIZE * BATCH_SIZE], Dtype(0.0001));
				}
			}
		}

		Blob<Dtype>* const blob_bottom_data_;
		Blob<Dtype>* const blob_bottom_label_;
		Blob<Dtype>* const blob_top_loss_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(DiceLossLayerTest, TestDtypesAndDevices);

	TYPED_TEST(DiceLossLayerTest, TestForward) {
		this->TestForward();
	}

	TYPED_TEST(DiceLossLayerTest, TestBackward) {
		this->TestBackward();
	}

} // namespace caffe