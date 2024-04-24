//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_tensor, create_cube) {
  using namespace kuiper_infer;
  int32_t size = 27;
  std::vector<float> datas;
  for (int i = 0; i < size; ++i) {
    datas.push_back(float(i));
  }
  arma::Cube<float> cube(3, 3, 3);
  memcpy(cube.memptr(), datas.data(), size * sizeof(float));
  LOG(INFO) << cube;
}

TEST(test_tensor, create_1dtensor) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 4);
  Tensor<float> f2(4);
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f2.raw_shapes().size(), 1);

}

TEST(test_tensor, create_3dtensor) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  ASSERT_EQ(f1.shapes().size(), 3);
  ASSERT_EQ(f1.size(), 24);
}

TEST(test_tensor, get_infos) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  ASSERT_EQ(f1.channels(), 2);
  ASSERT_EQ(f1.rows(), 3);
  ASSERT_EQ(f1.cols(), 4);
}

TEST(test_tensor, tensor_init1D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4);
  f1.Fill(1.f);
  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);
  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}

TEST(test_tensor, tensor_init2D) {
  using namespace kuiper_infer;
  Tensor<float> f1(4, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_3) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;
  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_2) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 2, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;
  LOG(INFO) << "data cols: " << cols;
  f1.Show();
}

TEST(test_tensor, tensor_init3D_1) {
  using namespace kuiper_infer;
  Tensor<float> f1(1, 1, 3);
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;
  f1.Show();
}
