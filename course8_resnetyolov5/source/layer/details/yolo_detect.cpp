// MIT License
// Copyright (c) 2022 - 傅莘莘
// Source URL: https://github.com/zjhellofss/KuiperInfer
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Created by fss on 22-12-26.
#include "yolo_detect.hpp"
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

YoloDetectLayer::YoloDetectLayer(
    int32_t stages, int32_t num_classes, int32_t num_anchors,
    std::vector<float> strides, std::vector<arma::fmat> anchor_grids,
    std::vector<arma::fmat> grids,
    std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers)
    : Layer("yolo"),
      stages_(stages),
      num_classes_(num_classes),
      num_anchors_(num_anchors),
      strides_(std::move(strides)),
      anchor_grids_(std::move(anchor_grids)),
      grids_(std::move(grids)),
      conv_layers_(std::move(conv_layers)) {}

InferStatus YoloDetectLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
    std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the yolo detect layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  const uint32_t stages = stages_;
  const uint32_t classes_info = num_classes_ + 5;
  const uint32_t input_size = inputs.size();
  const uint32_t batch_size = outputs.size();

  if (input_size / batch_size != stages_ || input_size % batch_size != 0) {
    LOG(ERROR) << "The input and output tensor array size of the yolo detect "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  CHECK(!this->conv_layers_.empty() && this->conv_layers_.size() == stages)
          << "The yolo detect layer do not have appropriate number of convolution "
             "operations";

  std::vector<std::vector<std::shared_ptr<Tensor<float>>>> batches(stages);
  for (uint32_t i = 0; i < input_size; ++i) {
    const uint32_t index = i / batch_size;
    const auto &input_data = inputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the yolo detect layer has an "
                    "empty tensor "
                 << i << "th";
      return InferStatus::kInferFailedInputEmpty;
    }
    CHECK(index <= batches.size());
    batches.at(index).push_back(input_data);
  }

  std::vector<std::vector<sftensor>> stage_outputs(stages);
  for (uint32_t stage = 0; stage < stages; ++stage) {
    const std::vector<std::shared_ptr<Tensor<float>>> &stage_input =
        batches.at(stage);

    CHECK(stage_input.size() == batch_size)
            << "The number of stage input in the yolo detect layer should be equal "
               "to batch size";

    std::vector<std::shared_ptr<Tensor<float>>> stage_output(batch_size);
    const auto status =
        this->conv_layers_.at(stage)->Forward(stage_input, stage_output);

    CHECK(status == InferStatus::kInferSuccess)
            << "Convolution layers infer failed in the yolo detect layer, error "
               "code: "
            << int(status);
    CHECK(stage_output.size() == batch_size)
            << "The number of stage output in the yolo detect layer should be "
               "equal to batch size";
    stage_outputs.at(stage) = stage_output;
  }

  uint32_t concat_rows = 0;
  for (uint32_t stage = 0; stage < stages; ++stage) {
    const std::vector<sftensor> stage_output = stage_outputs.at(stage);
    const uint32_t nx = stage_output.front()->rows();
    const uint32_t ny = stage_output.front()->cols();
    for (uint32_t i = 0; i < stage_output.size(); ++i) {
      CHECK(stage_output.at(i)->rows() == nx &&
          stage_output.at(i)->cols() == ny);
    }

    std::shared_ptr<Tensor<float>> stages_tensor =
        this->stages_tensors_.at(stage);
    for (uint32_t b = 0; b < batch_size; ++b) {
      const std::shared_ptr<Tensor<float>> &input = stage_output.at(b);
      CHECK(input != nullptr && !input->empty());
      CHECK_EQ(input->rows(), nx);
      CHECK_EQ(input->cols(), ny);
      input->Reshape({stages, uint32_t(classes_info), ny * nx}, true);

      CHECK_EQ(stages_tensor->channels(), batch_size);
      CHECK_EQ(stages_tensor->rows(), stages_ * nx * ny);
      CHECK_EQ(stages_tensor->cols(), classes_info);

      const arma::fcube &input_data = input->data();
      const arma::fcube &input_data_exp = 1.f / (1.f + arma::exp(-input_data));

      arma::fmat &x_stages = stages_tensor->slice(b);
      for (uint32_t na = 0; na < num_anchors_; ++na) {
        x_stages.submat(ny * nx * na, 0, ny * nx * (na + 1) - 1,
                        classes_info - 1) = input_data_exp.slice(na).t();
      }

      const arma::fmat &xy = x_stages.submat(0, 0, x_stages.n_rows - 1, 1);
      const arma::fmat &wh = x_stages.submat(0, 2, x_stages.n_rows - 1, 3);
      x_stages.submat(0, 0, x_stages.n_rows - 1, 1) =
          (xy * 2 + grids_[stage]) * strides_[stage];
      x_stages.submat(0, 2, x_stages.n_rows - 1, 3) =
          arma::pow((wh * 2), 2) % anchor_grids_[stage];
    }
    concat_rows += stages_tensor->rows();
  }

  uint32_t current_rows = 0;
  arma::fcube f1(concat_rows, classes_info, batch_size);
  for (std::shared_ptr<ftensor> stages_tensor : this->stages_tensors_) {
    f1.subcube(current_rows, 0, 0, current_rows + stages_tensor->rows() - 1,
               classes_info - 1, batch_size - 1) = stages_tensor->data();
    current_rows += stages_tensor->rows();
  }

  for (int i = 0; i < f1.n_slices; ++i) {
    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(1, concat_rows, classes_info);
      outputs.at(i) = output;
    }
    CHECK(output->rows() == f1.slice(i).n_rows);
    CHECK(output->cols() == f1.slice(i).n_cols);
    output->slice(0) = std::move(f1.slice(i));
  }
  return InferStatus::kInferSuccess;
}

void YoloDetectLayer::set_stage_tensors(
    const std::vector<sftensor> &stage_tensors) {
  this->stages_tensors_ = stage_tensors;
}

ParseParameterAttrStatus YoloDetectLayer::GetInstance(
    const std::shared_ptr<RuntimeOperator> &op,
    std::shared_ptr<Layer> &yolo_detect_layer) {
  CHECK(op != nullptr) << "Yolo detect operator is nullptr";

  const auto &attrs = op->attribute;
  CHECK(!attrs.empty()) << "Operator attributes is empty!";

  if (attrs.find("pnnx_5") == attrs.end()) {
    LOG(ERROR) << "Can not find the in yolo strides attribute";
    return ParseParameterAttrStatus::kAttrMissingYoloStrides;
  }

  const auto &stages_attr = attrs.at("pnnx_5");
  if (stages_attr->shape.empty()) {
    LOG(ERROR) << "Can not find the in yolo strides attribute";
    return ParseParameterAttrStatus::kAttrMissingYoloStrides;
  }

  int stages_number = stages_attr->shape.at(0);
  CHECK(stages_number == 3) << "Only support three stages yolo detect head";
  CHECK(op->input_operands_seq.size() == stages_number);

  std::vector<float> strides = stages_attr->get<float>();
  CHECK(strides.size() == stages_number)
          << "Stride number is not equal to strides";

  int32_t num_anchors = -1;
  std::vector<arma::fmat> anchor_grids;
  for (int i = 4; i >= 0; i -= 2) {
    const std::string &pnnx_name = "pnnx_" + std::to_string(i);
    const auto &anchor_grid_attr = attrs.find(pnnx_name);
    if (anchor_grid_attr == attrs.end()) {
      LOG(ERROR) << "Can not find the in yolo anchor grides attribute";
      return ParseParameterAttrStatus::kAttrMissingYoloAnchorGrides;
    }
    const auto &anchor_grid = anchor_grid_attr->second;
    const auto &anchor_shapes = anchor_grid->shape;
    const std::vector<float> &anchor_weight_data = anchor_grid->get<float>();
    CHECK(!anchor_shapes.empty() && anchor_shapes.size() == 5 &&
        anchor_shapes.front() == 1)
            << "Anchor shape has a wrong size";

    if (num_anchors == -1) {
      num_anchors = anchor_shapes.at(1);
      CHECK(num_anchors > 0) << "The number of anchors must greater than zero";
    } else {
      CHECK(num_anchors == anchor_shapes.at(1))
              << "The number of anchors must be the same";
    }

    const uint32_t anchor_rows =
        anchor_shapes.at(1) * anchor_shapes.at(2) * anchor_shapes.at(3);
    const uint32_t anchor_cols = anchor_shapes.at(4);
    CHECK(anchor_weight_data.size() == anchor_cols * anchor_rows)
            << "Anchor weight has a wrong size";

    arma::fmat anchor_grid_matrix(anchor_weight_data.data(), anchor_cols,
                                  anchor_rows);
    anchor_grids.emplace_back(anchor_grid_matrix.t());
  }

  std::vector<arma::fmat> grids;
  std::vector<int32_t> grid_indexes{6, 3, 1};
  for (const auto grid_index : grid_indexes) {
    const std::string &pnnx_name = "pnnx_" + std::to_string(grid_index);
    const auto &grid_attr = attrs.find(pnnx_name);
    if (grid_attr == attrs.end()) {
      LOG(ERROR) << "Can not find the in yolo grides attribute";
      return ParseParameterAttrStatus::kAttrMissingYoloGrides;
    }

    const auto &grid = grid_attr->second;
    const auto &shapes = grid->shape;
    const std::vector<float> &weight_data = grid->get<float>();
    CHECK(!shapes.empty() && shapes.size() == 5 && shapes.front() == 1)
            << "Grid shape has a wrong size";

    if (num_anchors == -1) {
      num_anchors = shapes.at(1);
      CHECK(num_anchors > 0) << "The number of anchors must greater than zero";
    } else {
      CHECK(num_anchors == shapes.at(1))
              << "The number of anchors must be the same";
    }

    const uint32_t grid_rows = shapes.at(1) * shapes.at(2) * shapes.at(3);
    const uint32_t grid_cols = shapes.at(4);
    CHECK(weight_data.size() == grid_cols * grid_rows)
            << "Grid weight has a wrong size";

    arma::fmat matrix(weight_data.data(), grid_cols, grid_rows);
    grids.emplace_back(matrix.t());
  }

  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers(stages_number);
  int32_t num_classes = -1;
  std::vector<sftensor> stage_tensors;
  for (int i = 0; i < stages_number; ++i) {
    const std::string &weight_name = "m." + std::to_string(i) + ".weight";
    if (attrs.find(weight_name) == attrs.end()) {
      LOG(ERROR) << "Can not find the in weight attribute " << weight_name;
      return ParseParameterAttrStatus::kAttrMissingWeight;
    }

    const auto &conv_attr = attrs.at(weight_name);
    const auto &out_shapes = conv_attr->shape;
    CHECK(out_shapes.size() == 4) << "Stage output shape must equal to four";

    const int out_channels = out_shapes.at(0);
    if (num_classes == -1) {
      CHECK(out_channels % num_anchors == 0)
              << "The number of output channel is wrong, it should divisible by "
                 "number of anchors";
      num_classes = out_channels / num_anchors - 5;
      CHECK(num_classes > 0)
              << "The number of object classes must greater than zero";
    }
    const int in_channels = out_shapes.at(1);
    const int kernel_h = out_shapes.at(2);
    const int kernel_w = out_shapes.at(3);

    CHECK_EQ(op->input_operands_seq.at(i)->shapes.size(), 4);
    const uint32_t batch_size = op->input_operands_seq.at(i)->shapes.front();

    const uint32_t nx = op->input_operands_seq.at(i)->shapes.at(2);
    const uint32_t ny = op->input_operands_seq.at(i)->shapes.at(3);

    auto stages_tensor = TensorCreate(batch_size, stages_number * nx * ny,
                                      uint32_t(num_classes + 5));
    stage_tensors.push_back(stages_tensor);

    conv_layers.at(i) = std::make_shared<ConvolutionLayer>(
        out_channels, in_channels, kernel_h, kernel_w, 0, 0, 1, 1, 1);
    const std::vector<float> &weights = conv_attr->get<float>();
    conv_layers.at(i)->set_weights(weights);

    const std::string &bias_name = "m." + std::to_string(i) + ".bias";
    if (attrs.find(bias_name) == attrs.end()) {
      LOG(ERROR) << "Can not find the in bias attribute";
      return ParseParameterAttrStatus::kAttrMissingBias;
    }
    const auto &bias_attr = attrs.at(bias_name);
    const std::vector<float> &bias = bias_attr->get<float>();
    conv_layers.at(i)->set_bias(bias);
  }

  yolo_detect_layer = std::make_shared<YoloDetectLayer>(
      stages_number, num_classes, num_anchors, std::move(strides),
      std::move(anchor_grids), std::move(grids), std::move(conv_layers));
  auto yolo_detect_layer_ =
      std::dynamic_pointer_cast<YoloDetectLayer>(yolo_detect_layer);

  yolo_detect_layer_->set_stage_tensors(stage_tensors);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kYoloGetInstance("models.yolo.Detect",
                                        YoloDetectLayer::GetInstance);

}  // namespace kuiper_infer