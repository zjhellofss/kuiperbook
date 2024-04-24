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

#ifndef KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#define KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
#include "convolution.hpp"
#include "layer/abstract/layer.hpp"

namespace kuiper_infer {
class YoloDetectLayer : public Layer {
 public:
  explicit YoloDetectLayer(
      int32_t stages, int32_t num_classes, int32_t num_anchors,
      std::vector<float> strides, std::vector<arma::fmat> anchor_grids,
      std::vector<arma::fmat> grids,
      std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers);

  InferStatus Forward(
      const std::vector<std::shared_ptr<Tensor<float>>>& inputs,
      std::vector<std::shared_ptr<Tensor<float>>>& outputs) override;

  static ParseParameterAttrStatus GetInstance(
      const std::shared_ptr<RuntimeOperator>& op,
      std::shared_ptr<Layer>& yolo_detect_layer);

  void set_stage_tensors(const std::vector<sftensor>& stage_tensors);

 private:
  int32_t stages_ = 0;
  int32_t num_classes_ = 0;
  int32_t num_anchors_ = 0;
  std::vector<float> strides_;
  std::vector<arma::fmat> anchor_grids_;
  std::vector<arma::fmat> grids_;
  std::vector<std::shared_ptr<ConvolutionLayer>> conv_layers_;
  std::vector<sftensor> stages_tensors_;
};
}  // namespace kuiper_infer
#endif  // KUIPER_INFER_SOURCE_LAYER_DETAILS_YOLO_DETECT_HPP_
