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

// Created by fss on 22-11-13.

#include "softmax.hpp"
#include <glog/logging.h>
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"
#include "utils/math/fmath.hpp"
namespace kuiper_infer {
#define POS_INDEX(outer_size, inner_size, axis_size) \
  outer_size* axis_sizes* inner_sizes + axis_size* inner_sizes + inner_size;

SoftmaxLayer::SoftmaxLayer(int dim)
    : NonParamLayer("Softmax"), softmax_dim_(dim) {}

InferStatus SoftmaxLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the softmax layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the softmax layer "
                  "do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  const uint32_t batch_size = inputs.size();
  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    CHECK(input != nullptr && !input->empty())
        << "The input tensor array in the softmax layer has an empty tensor "
        << i << " th";

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    if (output == nullptr || output->empty()) {
      output = std::make_shared<Tensor<float>>(input->shapes());
      outputs.at(i) = output;
    }
    CHECK(input->shapes() == output->shapes())
        << "The input and output tensor shapes of the softmax layer do not "
           "match "
        << i << " th";
    int dim = this->softmax_dim_;
    std::vector<uint32_t> raw_shapes = input->raw_shapes();

    if (dim < 0) {
      dim += int(raw_shapes.size());
    }

    if (dim < 0 || dim >= 3 || dim > raw_shapes.size()) {
      LOG(FATAL) << "Error softmax dimension, which need between 0 and 2, "
                    "but dimension is "
                 << dim;
    }
    const uint32_t padding_size_num = 3 - raw_shapes.size();
    for (uint32_t j = 0; j < padding_size_num; ++j) {
      raw_shapes.push_back(1);
    }

    /**
     * [...(inner size) dim ...(outer_size)
     * 将输入的数据按dim维度拆分为两部分，分别为inner和outer
     * 开始位置到dim轴位置的数据量是inner_size,
     * dim轴位置到结束位置的数据量是outer_sizes
     */
    const uint32_t inner_sizes = std::accumulate(
        raw_shapes.begin() + dim + 1, raw_shapes.end(), 1, std::multiplies());
    const uint32_t outer_sizes = std::accumulate(
        raw_shapes.begin(), raw_shapes.begin() + dim, 1, std::multiplies());

    // dim轴数据的数量
    const uint32_t axis_sizes = raw_shapes.at(dim);
    CHECK_EQ(axis_sizes * outer_sizes * inner_sizes, input->size());

    const auto& input_values = input->values(true);
    std::vector<float> output_values(input_values.size());
    for (uint32_t outer_size = 0; outer_size < outer_sizes; ++outer_size) {
      for (uint32_t inner_size = 0; inner_size < inner_sizes; ++inner_size) {
        float max_value = std::numeric_limits<float>::lowest();
        // 迭代当前dim中的数据，并找到其中的最大值
        for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
          uint32_t index = POS_INDEX(outer_size, inner_size, axis_size);
          float cur_value = input_values.at(index);
          if (cur_value > max_value) {
            max_value = cur_value;
          }
        }

        float sum_value = 0.f;
        // 迭代当前dim中的数据，并进行求和
        for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
          uint32_t index = POS_INDEX(outer_size, inner_size, axis_size);
          float cur_value = input_values.at(index);
          float exp_sub_value = fmath::exp(cur_value - max_value);

          sum_value += exp_sub_value;
          output_values.at(index) = exp_sub_value;
        }

        // 迭代当前dim中的数据，求exp(cur_value - max_value) / sum_value
        for (uint32_t axis_size = 0; axis_size < axis_sizes; ++axis_size) {
          uint32_t index = POS_INDEX(outer_size, inner_size, axis_size);

          float exp_sub_value = output_values.at(index);
          output_values.at(index) = exp_sub_value / sum_value;
        }
      }
    }
    output->Fill(output_values, true);
  }
  return InferStatus::kInferSuccess;
}
ParseParameterAttrStatus SoftmaxLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& softmax_layer) {
  CHECK(op != nullptr) << "SoftMax operator is nullptr";
  const auto& params = op->params;
  if (params.find("dim") == params.end()) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  auto dim_param = params.at("dim");
  if (dim_param == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  auto dim = std::dynamic_pointer_cast<RuntimeParameterInt>(dim_param);
  if (dim == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }
  softmax_layer = std::make_shared<SoftmaxLayer>(dim->value);  // 创建softmax层
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}
LayerRegistererWrapper kSoftMaxCreateInstanceNN("nn.Softmax",
                                             SoftmaxLayer::CreateInstance);
LayerRegistererWrapper kSoftMaxCreateInstanceF("F.softmax",
                                            SoftmaxLayer::CreateInstance);
}  // namespace kuiper_infer