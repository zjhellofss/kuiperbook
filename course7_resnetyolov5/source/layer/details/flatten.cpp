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

// Created by fss on 22-12-9.
#include "flatten.hpp"
#include <numeric>
#include "data/tensor_util.hpp"
#include "layer/abstract/layer_factory.hpp"

namespace kuiper_infer {
FlattenLayer::FlattenLayer(int start_dim, int end_dim)
    : NonParamLayer("Flatten"), start_dim_(start_dim), end_dim_(end_dim) {}

InferStatus FlattenLayer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  if (inputs.empty()) {
    LOG(ERROR) << "The input tensor array in the flatten layer is empty";
    return InferStatus::kInferFailedInputEmpty;
  }

  if (inputs.size() != outputs.size()) {
    LOG(ERROR) << "The input and output tensor array size of the flatten "
                  "layer do not match";
    return InferStatus::kInferFailedInputOutSizeMatchError;
  }

  int start_dim = start_dim_;
  int end_dim = end_dim_;
  int total_dims = 4;  // NCHW

  if (start_dim < 0) {
    start_dim = total_dims + start_dim;
  }
  if (end_dim < 0) {
    end_dim = total_dims + end_dim;
  }

  CHECK(end_dim > start_dim) << "The end dim must greater than start dim";
  CHECK(end_dim <= 3 && start_dim >= 1)
      << "The end dim must less than two and start dim must greater than zero";

  const uint32_t batch_size = inputs.size();

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::shared_ptr<Tensor<float>>& input = inputs.at(i);
    if (input == nullptr || input->empty()) {
      LOG(ERROR) << "The input tensor array in the flatten layer has"
                    " an empty tensor "
                 << i << " th";
      return InferStatus::kInferFailedInputEmpty;
    }

    auto shapes = input->shapes();
    shapes.insert(shapes.begin(), batch_size);
    uint32_t elements_size =
        std::accumulate(shapes.begin() + start_dim,
                        shapes.begin() + end_dim + 1, 1, std::multiplies());

    std::shared_ptr<Tensor<float>> output = outputs.at(i);
    output = TensorClone(input);
    CHECK(input->size() == output->size())
        << "The output and input shapes of the flatten layer do "
           "not match "
        << i << " th";
    outputs.at(i) = output;

    if (start_dim == 1 && end_dim == 3) {
      output->Reshape({elements_size}, true);
    } else if (start_dim == 2 && end_dim == 3) {
      uint32_t channels = input->channels();
      output->Reshape({channels, elements_size}, true);
    } else if (start_dim == 1 && end_dim == 2) {
      uint32_t cols = input->cols();
      output->Reshape({elements_size, cols}, true);
    } else {
      LOG(FATAL) << "Wrong flatten dim: "
                 << "start dim: " << start_dim << " end dim: " << end_dim;
    }
  }
  return InferStatus::kInferSuccess;
}

ParseParameterAttrStatus FlattenLayer::CreateInstance(
    const std::shared_ptr<RuntimeOperator>& op,
    std::shared_ptr<Layer>& flatten_layer) {
  CHECK(op != nullptr) << "Flatten operator is nullptr";
  const auto& params = op->params;

  if (params.find("end_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  if (params.find("start_dim") == params.end()) {
    LOG(ERROR) << "Can not find the dimension parameter";
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  auto start_dim =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("start_dim"));

  auto end_dim =
      std::dynamic_pointer_cast<RuntimeParameterInt>(params.at("end_dim"));

  if (start_dim == nullptr || end_dim == nullptr) {
    return ParseParameterAttrStatus::kParameterMissingDim;
  }

  flatten_layer =
      std::make_shared<FlattenLayer>(start_dim->value, end_dim->value);
  return ParseParameterAttrStatus::kParameterAttrParseSuccess;
}

LayerRegistererWrapper kFlattenCreateInstance("torch.flatten",
                                           FlattenLayer::CreateInstance);

}  // namespace kuiper_infer