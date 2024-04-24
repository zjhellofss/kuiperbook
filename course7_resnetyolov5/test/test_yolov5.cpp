#include <glog/logging.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "data/tensor.hpp"
#include "image_util.hpp"
#include "runtime/runtime_ir.hpp"
#include <gtest/gtest.h>
#include <vector>

kuiper_infer::sftensor PreProcessImage(const cv::Mat &image,
                                       const int32_t input_h,
                                       const int32_t input_w) {
  assert(!image.empty());
  using namespace kuiper_infer;
  const int32_t input_c = 3;

  int stride = 32;
  cv::Mat out_image;
  Letterbox(image, out_image, {input_h, input_w}, stride, {114, 114, 114},
            true);

  cv::Mat rgb_image;
  cv::cvtColor(out_image, rgb_image, cv::COLOR_BGR2RGB);

  cv::Mat normalize_image;
  rgb_image.convertTo(normalize_image, CV_32FC3, 1. / 255.);

  std::vector<cv::Mat> split_images;
  cv::split(normalize_image, split_images);
  assert(split_images.size() == input_c);

  std::shared_ptr<Tensor<float>> input =
      std::make_shared<Tensor<float>>(input_c, input_h, input_w);
  input->Fill(0.f);

  int index = 0;
  int offset = 0;
  for (const auto &split_image : split_images) {
    assert(split_image.total() == input_w * input_h);
    const cv::Mat &split_image_t = split_image.t();
    memcpy(input->slice(index).memptr(), split_image_t.data,
           sizeof(float) * split_image.total());
    index += 1;
    offset += split_image.total();
  }
  return input;
}

void YoloDemo(const std::vector<std::string> &image_paths,
              const std::string &param_path, const std::string &bin_path,
              const uint32_t batch_size, const float conf_thresh = 0.25f,
              const float iou_thresh = 0.25f) {
  using namespace kuiper_infer;
  const int32_t input_h = 640;
  const int32_t input_w = 640;

  RuntimeGraph graph(param_path, bin_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  assert(batch_size == image_paths.size());
  std::vector<sftensor> inputs;
  for (uint32_t i = 0; i < batch_size; ++i) {
    const auto &input_image = cv::imread(image_paths.at(i));
    sftensor input = PreProcessImage(input_image, input_h, input_w);
    assert(input->rows() == 640);
    assert(input->cols() == 640);
    inputs.push_back(input);
  }

  std::vector<std::shared_ptr<Tensor<float>>> outputs;

  outputs = graph.Forward(inputs, true);
  assert(outputs.size() == inputs.size());
  assert(outputs.size() == batch_size);

  for (int i = 0; i < outputs.size(); ++i) {
    const auto &image = cv::imread(image_paths.at(i));
    const int32_t origin_input_h = image.size().height;
    const int32_t origin_input_w = image.size().width;

    const auto &output = outputs.at(i);
    assert(!output->empty());
    const auto &shapes = output->shapes();
    assert(shapes.size() == 3);

    const uint32_t elements = shapes.at(1);
    const uint32_t num_info = shapes.at(2);
    std::vector<Detection> detections;

    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;

    const uint32_t b = 0;
    for (uint32_t e = 0; e < elements; ++e) {
      float cls_conf = output->at(b, e, 4);
      if (cls_conf >= conf_thresh) {
        int center_x = (int) (output->at(b, e, 0));
        int center_y = (int) (output->at(b, e, 1));
        int width = (int) (output->at(b, e, 2));
        int height = (int) (output->at(b, e, 3));
        int left = center_x - width / 2;
        int top = center_y - height / 2;

        int best_class_id = -1;
        float best_conf = -1.f;
        for (uint32_t j = 5; j < num_info; ++j) {
          if (output->at(b, e, j) > best_conf) {
            best_conf = output->at(b, e, j);
            best_class_id = int(j - 5);
          }
        }

        boxes.emplace_back(left, top, width, height);
        confs.emplace_back(best_conf * cls_conf);
        class_ids.emplace_back(best_class_id);
      }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, conf_thresh, iou_thresh, indices);

    for (int idx : indices) {
      Detection det;
      det.box = cv::Rect(boxes[idx]);
      ScaleCoords(cv::Size{input_w, input_h}, det.box,
                  cv::Size{origin_input_w, origin_input_h});

      det.conf = confs[idx];
      det.class_id = class_ids[idx];
      detections.emplace_back(det);
    }

    int font_face = cv::FONT_HERSHEY_COMPLEX;
    double font_scale = 2;

    for (const auto &detection : detections) {
      cv::rectangle(image, detection.box, cv::Scalar(255, 255, 255), 4);
      cv::putText(image, std::to_string(detection.class_id),
                  cv::Point(detection.box.x, detection.box.y), font_face,
                  font_scale, cv::Scalar(255, 255, 0), 4);
    }
    cv::imwrite(std::string("output") + std::to_string(i) + ".jpg", image);
  }
}

TEST(test_network, yolov5) {
  const uint32_t batch_size = 1;
  std::vector<std::string> image_paths;

  for (uint32_t i = 0; i < batch_size; ++i) {
    const std::string &image_path = "./course7_resnetyolov5/model_file/car.jpg";  // 可以放不同的图片
    image_paths.push_back(image_path);
  }
  const std::string &param_path = "course7_resnetyolov5/model_file/yolov5s.pnnx.param";
  const std::string &bin_path = "course7_resnetyolov5/model_file/yolov5s.pnnx.bin";

  YoloDemo(image_paths, param_path, bin_path, batch_size);
}