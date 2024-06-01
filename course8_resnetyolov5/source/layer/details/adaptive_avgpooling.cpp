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

// Created by fss on 22-11-12.
#include "adaptive_avgpooling.hpp"
#include <glog/logging.h>
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {

AdaptiveAveragePoolingLayer::AdaptiveAveragePoolingLayer(uint32_t output_h,
                                                         uint32_t output_w)
    : NonParamLayer("AdaptiveAveragePooling"),
      output_h_(output_h),
      output_w_(output_w) {
  CHECK_GT(output_h_, 0);
  CHECK_GT(output_w_, 0);
}

InferStatus AdaptiveAveragePoolingLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR)
        << "The input tensor array in the adaptive pooling layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the adaptive "
                  "pooling layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch = inputs.size();
  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<ftensor>& input_data = inputs.at(i);
    const std::shared_ptr<ftensor>& output_data = outputs.at(i);
    if (input_data == nullptr || input_data->empty()) {
      LOG(ERROR) << "The input tensor array in the adaptive pooling layer has "
                    "an empty tensor "
                 << i << "th";
      return InferStatus::kInferFailedInputEmpty;
    }
    if (output_data != nullptr && !output_data->empty()) {
      if (output_data->rows() != output_h_ ||
          output_data->cols() != output_w_) {
        LOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                      "has an incorrectly sized tensor "
                   << i << "th";
        return InferStatus::kInferFailedOutputSizeError;
      }
    }
  }

  for (uint32_t i = 0; i < batch; ++i) {
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);
    CHECK(input_data != nullptr && !input_data->empty())
        << "The input tensor array in the adaptive pooling layer has an empty "
           "tensor "
        << i << "th";

    const uint32_t input_h = input_data->rows();
    const uint32_t input_w = input_data->cols();
    const uint32_t input_c = input_data->channels();
    const uint32_t stride_h = uint32_t(std::floor(input_h / output_h_));
    const uint32_t stride_w = uint32_t(std::floor(input_w / output_w_));
    CHECK(stride_w > 0 && stride_h > 0)
        << "The stride parameter is set incorrectly. It must always be greater "
           "than 0";

    const uint32_t pooling_h =
        (int)input_h - (int(output_h_) - 1) * int(stride_h);
    const uint32_t pooling_w =
        (int)input_w - (int(output_w_) - 1) * int(stride_w);
    CHECK(pooling_w > 0 && pooling_h > 0)
        << "The pooling parameter is set incorrectly. It must always be "
           "greater than 0";

    std::shared_ptr<Tensor<float>> output_data = outputs.at(i);
    if (output_data == nullptr || output_data->empty()) {
      DLOG(ERROR) << "The output tensor array in the adaptive pooling layer "
                     "has an empty tensor "
                  << i << "th";
      output_data =
          std::make_shared<Tensor<float>>(input_c, output_h_, output_w_);
      outputs.at(i) = output_data;
    }

    CHECK(output_data->rows() == output_h_ &&
          output_data->cols() == output_w_ &&
          output_data->channels() == input_c)
        << "The output tensor array in the adaptive pooling layer has an "
           "incorrectly sized tensor "
        << i << "th";

    const uint32_t pooling_size = pooling_h * pooling_w;
    for (uint32_t ic = 0; ic < input_c; ++ic) {
      const arma::fmat& input_channel = input_data->slice(ic);
      arma::fmat& output_channel = output_data->slice(ic);
      for (uint32_t c = 0; c < input_w - pooling_w + 1; c += stride_w) {
        int output_col = int(c / stride_w);
        for (uint32_t r = 0; r < input_h - pooling_h + 1; r += stride_h) {
          int output_row = int(r / stride_h);
          float mean_value = 0.f;
          float* output_channel_ptr = output_channel.colptr(output_col);
          for (uint32_t w = 0; w < pooling_w; ++w) {
            const float* col_ptr = input_channel.colptr(c + w) + r;
            for (uint32_t h = 0; h < pooling_h; ++h) {
              float current_value = *(col_ptr + h);
              mean_value = mean_value + current_value;
            }
          }
          *(output_channel_ptr + output_row) = mean_value / float(pooling_size);
        }
      }
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus AdaptiveAveragePoolingLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& avg_layer) {
  CHECK(op != nullptr) << "Adaptive pooling operator is nullptr";
  const auto& params = op->params;
  CHECK(!params.empty()) << "Operator parameter is empty";

  auto output_hw = std::dynamic_pointer_cast<RuntimeParameterIntArray>(
      params.at("output_size"));
  if (!output_hw) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }

  const auto& output_hw_arr = output_hw->value;
  if (output_hw_arr.size() != 2) {
    LOG(ERROR) << "Can not find the output size parameter";
    return ParseParameterAttrStatus::kParameterMissingOutHW;
  }
  avg_layer = std::make_shared<AdaptiveAveragePoolingLayer>(
      output_hw_arr.at(0), output_hw_arr.at(1));
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kAdaptiveAvgpoolingCreateInstance(
    "nn.AdaptiveAvgPool2d", AdaptiveAveragePoolingLayer::CreateInstance);

}  // namespace kuiper_infer