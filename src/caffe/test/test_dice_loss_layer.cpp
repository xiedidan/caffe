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

namespace caffe {
	template <typename TypeParam>
	class DiceLossLayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;

	protected:
		DiceLossLayerTest() :
			blob_bottom_data_(new Blob<Dtype>(4, 2, 1024, 1)),
			blob_bottom_label_(new Blob<Dtype>(4, 1, 1024, 1)),
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
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 1024; j++) {
					data[i * 1024 + j] = Dtype(1);
					data[i * 1024 + j + 4 * 1024] = Dtype(0);
					label[i * 1024 + j] = Dtype(1);
				}
			}

			LayerParameter layerParam;
			DiceLossLayer<Dtype> diceLossLayer(layerParam);

			diceLossLayer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
			diceLossLayer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

			const Dtype loss = blob_top_loss_->cpu_data()[0];
			EXPECT_NEAR(Dtype(4.0), loss, Dtype(0.0001));
		}

		void TestBackward() {
			Dtype* data = blob_bottom_data_->mutable_cpu_data();
			Dtype* label = blob_bottom_label_->mutable_cpu_data();
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 1024; j++) {
					data[i * 1024 + j] = Dtype(1);
					data[i * 1024 + j + 4 * 1024] = Dtype(0);
					label[i * 1024 + j] = Dtype(1);
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
			
			for (int i = 0; i < 1024; i++) {
				const Dtype* diff = this->blob_bottom_vec_[0]->cpu_diff();
				EXPECT_NEAR(Dtype(0), diff[i], Dtype(0.0001));
				EXPECT_NEAR(Dtype(0), diff[i + 1024], Dtype(0.0001));
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