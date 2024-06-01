//
// Created by fss on 23-6-25.
//
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

TEST(test_ir, topo) {
  using namespace kuiper_infer;
  std::string bin_path("course4_buildgraph/model_file/resnet18_batch1.pnnx.bin");
  std::string param_path("course4_buildgraph/model_file/resnet18_batch1.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "Index: " << index << " Type: " << operator_->type
              << " Name: " << operator_->name;
    index += 1;
  }
}

TEST(test_ir, build_output_ops) {
  using namespace kuiper_infer;
  std::string bin_path("course4_buildgraph/model_file/simple_ops.pnnx.bin");
  std::string param_path("course4_buildgraph/model_file/simple_ops.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "Index: " << index << " Name: " << operator_->name;
    index += 1;
  }
}

TEST(test_ir, build_output_ops2) {
  using namespace kuiper_infer;
  std::string bin_path("course4_buildgraph/model_file/simple_ops.pnnx.bin");
  std::string param_path("course4_buildgraph/model_file/simple_ops.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  const auto &topo_queues = graph.get_topo_queues();

  int index = 0;
  for (const auto &operator_ : topo_queues) {
    LOG(INFO) << "operator name: " << operator_->name;
    for (const auto &pair : operator_->output_operators) {
      LOG(INFO) << "output: " << pair.first;
    }
    LOG(INFO) << "-------------------------";
    index += 1;
  }
}

TEST(test_ir, build1_status) {
  using namespace kuiper_infer;
  std::string bin_path("course4_buildgraph/model_file/simple_ops.pnnx.bin");
  std::string param_path("course4_buildgraph/model_file/simple_ops.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  ASSERT_EQ(int(graph.graph_state()), -2);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  ASSERT_EQ(int(graph.graph_state()), -1);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  ASSERT_EQ(int(graph.graph_state()), 0);
}

TEST(test_ir, runtime_graph_output_init1) {
  using namespace kuiper_infer;
  std::vector<pnnx::Operator *> pnnx_operators;
  std::vector<std::shared_ptr<RuntimeOperator>> run_ops;
  for (int i = 0; i < 4; ++i) {
    pnnx::Operator *pnnx_op = new pnnx::Operator;
    pnnx::Operand *pnnx_number = new pnnx::Operand;
    pnnx_number->type = 1;
    pnnx_number->shape = std::vector<int>{8, 3, 32, 32};
    pnnx_op->outputs.push_back(pnnx_number);
    pnnx_operators.push_back(pnnx_op);
    run_ops.push_back(std::make_shared<RuntimeOperator>());
  }

  RuntimeOperatorUtils::InitOperatorOutput(pnnx_operators, run_ops);

  for (const auto &run_op : run_ops) {
    const auto &output_datas = run_op->output_operands;
    ASSERT_EQ(output_datas->shapes.size(), 4);
    ASSERT_EQ(output_datas->datas.size(), 8);
    for (const auto &output_data : output_datas->datas) {
      const auto &raw_shapes = output_data->raw_shapes();
      ASSERT_EQ(raw_shapes.size(), 3);
      ASSERT_EQ(raw_shapes.at(0), 3);
      ASSERT_EQ(raw_shapes.at(1), 32);
      ASSERT_EQ(raw_shapes.at(2), 32);

      ASSERT_EQ(output_data->rows(), 32);
      ASSERT_EQ(output_data->cols(), 32);
      ASSERT_EQ(output_data->channels(), 3);
      output_data->data().resize(32, 16, 6);
    }
  }
}

TEST(test_ir, build1_output_tensors) {
  using namespace kuiper_infer;
  std::string bin_path("course4_buildgraph/model_file/simple_ops2.pnnx.bin");
  std::string param_path("course4_buildgraph/model_file/simple_ops2.pnnx.param");
  RuntimeGraph graph(param_path, bin_path);
  ASSERT_EQ(int(graph.graph_state()), -2);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);
  ASSERT_EQ(int(graph.graph_state()), -1);
  graph.Build("pnnx_input_0", "pnnx_output_0");
  ASSERT_EQ(int(graph.graph_state()), 0);

  const auto &ops = graph.operators();
  for (const auto &op : ops) {
    LOG(INFO) << op->name;
    // 打印op输出空间的张量
    const auto &operand = op->output_operands;
    if (operand->datas.empty()) {
      continue;
    }
    const uint32_t batch_size = operand->datas.size();
    LOG(INFO) << "batch: " << batch_size;

    for (uint32_t i = 0; i < batch_size; ++i) {
      const auto &data = operand->datas.at(i);
      LOG(INFO) << "channel: " << data->channels()
                << " height: " << data->rows() << " cols: " << data->cols();
    }
  }
}
