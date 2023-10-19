/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/reduce_scatter_combiner.h"

#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

constexpr int64_t kMaxCombineCount = 256;
constexpr int64_t kMaxByteCount = 10 * 1024 * 1024;

int64_t ReduceScatterCount(const HloModule& module) {
  int64_t count = 0;
  for (HloInstruction* hlo : module.entry_computation()->instructions()) {
    if (hlo->opcode() == HloOpcode::kReduceScatter ||
        (hlo->opcode() == HloOpcode::kAsyncStart &&
         hlo->async_wrapped_instruction()->opcode() ==
             HloOpcode::kReduceScatter)) {
      ++count;
    }
  }
  return count;
}

class ReduceScatterCombinerTest : public HloTestBase {
 public:
  StatusOr<std::unique_ptr<HloModule>> RunPass(
      absl::string_view hlo_module, bool expect_change,
      int64_t byte_threshold = kMaxByteCount,
      int64_t count_threshold = kMaxCombineCount, bool combine_by_dim = true) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_module));
    auto changed =
        ReduceScatterCombiner(byte_threshold, count_threshold, combine_by_dim)
            .Run(module.get());
    if (!changed.ok()) {
      return changed.status();
    }
    EXPECT_EQ(changed.value(), expect_change);
    return StatusOr<std::unique_ptr<HloModule>>(std::move(module));
  }
};

TEST_F(ReduceScatterCombinerTest, Simple) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(*module), 1);
}

TEST_F(ReduceScatterCombinerTest, SimpleMultipleGroups) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8, 8] parameter(1)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4, 8] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs2 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs3 = f32[8, 4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[4, 8], f32[8, 4], f32[8, 4])
      tuple(rs0, rs1, rs2, rs3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/true));
  EXPECT_EQ(ReduceScatterCount(*module), 2);
}

TEST_F(ReduceScatterCombinerTest, DifferentDimensions) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  p1 = f32[8, 8] parameter(1)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4, 8] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs2 = f32[8, 4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  rs3 = f32[8, 4] reduce-scatter(p1), replica_groups={{0,1}}, dimensions={1},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[4, 8], f32[8, 4], f32[8, 4])
      tuple(rs0, rs1, rs2, rs3)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, RunPass(hlo_string, /*expect_change=*/true, kMaxByteCount,
                           kMaxCombineCount, /*combine_by_dim=*/false));
  EXPECT_EQ(ReduceScatterCount(*module), 1);
}

// Test that dependent reduce-scatter do not get combined.
TEST_F(ReduceScatterCombinerTest, DependentReduceScatter) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8, 8] parameter(0)
  rs0 = f32[4, 8] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[2, 8] reduce-scatter(rs0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4, 8], f32[2, 8]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterCombinerTest, DoNotCombineMismatched) {
  absl::string_view hlo_string = R"(
HloModule m

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  rs0 = f32[4] reduce-scatter(p0), replica_groups={{0,1}}, dimensions={0},
      to_apply=sum
  rs1 = f32[4] reduce-scatter(p1), replica_groups={{1,0}}, dimensions={0},
      to_apply=sum
  ROOT t = (f32[4], f32[4]) tuple(rs0, rs1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

TEST_F(ReduceScatterCombinerTest, DoNotCombineWithoutReductionKind) {
  absl::string_view hlo_string = R"(
HloModule TestModule

region_0 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

region_1 {
  Arg_1 = bf16[] parameter(1)
  Arg_0 = bf16[] parameter(0)
  convert_1 = f32[] convert(Arg_1)
  convert_0 = f32[] convert(Arg_0)
  add0 = f32[] add(convert_1, convert_0)
  ROOT convert_2 = bf16[] convert(add0)
}

ENTRY entry{
  param0 = bf16[512,256]{1,0} parameter(0)
  param1 = bf16[512,256]{1,0} parameter(1)
  reduce-scatter.0 = bf16[512,256]{1,0} reduce-scatter(param0),
      replica_groups={{0}}, dimensions={0}, to_apply=region_0
  reduce-scatter.1 = bf16[512,256]{1,0} reduce-scatter(param1),
      replica_groups={{0}}, dimensions={0}, to_apply=region_1
  ROOT add.0 = tuple(reduce-scatter.0, reduce-scatter.1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          RunPass(hlo_string, /*expect_change=*/false));
}

using AsyncReduceScatterCombinerTest = HloTestBase;

TEST_F(AsyncReduceScatterCombinerTest, Simple) {
  const absl::string_view hlo_string = R"(
HloModule m, is_scheduled=true

sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add.2 = f32[] add(a, b)
}

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  p2 = f32[8] parameter(2)
  p3 = f32[8] parameter(3)
  rs0 = ((f32[8]), f32[4])  reduce-scatter-start(p0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs0d = f32[4] reduce-scatter-done(rs0), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1 = ((f32[8]), f32[4])  reduce-scatter-start(p1), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs1d = f32[4] reduce-scatter-done(rs1), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  foo = f32[4] add(rs0d, rs1d)
  rs2 = ((f32[8]), f32[4])  reduce-scatter-start(p2), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs2d = f32[4] reduce-scatter-done(rs2), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs3 = ((f32[8]), f32[4])  reduce-scatter-start(p3), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  rs3d = f32[4] reduce-scatter-done(rs3), replica_groups={{0,1}}, dimensions={0}, to_apply=sum
  ROOT t = (f32[4], f32[4], f32[4], f32[4]) tuple(rs0d, rs1d, rs2d, rs3d)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  VLOG(2) << "before: " << module->ToString();
  AsyncReduceScatterCombiner combine;
  ASSERT_EQ(ReduceScatterCount(*module), 4);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  VLOG(1) << "after: " << module->ToString();
  EXPECT_TRUE(changed);
  EXPECT_EQ(ReduceScatterCount(*module), 2);
}

}  // namespace
}  // namespace xla
