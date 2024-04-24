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

// Created by fss on 22-11-15.
#include "layer/abstract/layer.hpp"
namespace kuiper_infer {

const std::vector<std::shared_ptr<Tensor<float>>>& Layer::weights() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

const std::vector<std::shared_ptr<Tensor<float>>>& Layer::bias() const {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<float>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_bias(const std::vector<std::shared_ptr<Tensor<float>>>& bias) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(const std::vector<float>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_weights(
    const std::vector<std::shared_ptr<Tensor<float>>>& weights) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

StatusCode Layer::Forward(
    const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
    std::vector<std::shared_ptr<Tensor<float>>>& outputs) {
  LOG(FATAL) << this->layer_name_ << " layer not implement yet!";
}

void Layer::set_runtime_operator(
    const std::shared_ptr<RuntimeOperator>& runtime_operator) {
  CHECK(runtime_operator != nullptr);
  this->runtime_operator_ = runtime_operator;
}

}  // namespace kuiper_infer