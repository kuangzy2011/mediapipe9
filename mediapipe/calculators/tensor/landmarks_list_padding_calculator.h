//[CTOUCH]
// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MEDIAPIPE_CALCULATORS_LANDMARKS_LIST_PADDING_CALCULATOR_H_
#define MEDIAPIPE_CALCULATORS_LANDMARKS_LIST_PADDING_CALCULATOR_H_

#include <memory>

#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"

namespace mediapipe {
namespace api2 {

// A calculator for converting landmars into a Tensor.
//
// Input:
//   LANDMARKS - std::vector<LandmarkList>
//     Landmarks to be converted into a Tensor.
//
// Output:
//   TENSORS - std::vector<Tensor>
//     Vector containing a single Tensor populated with landmark values.
//
// Example:
// node {
//   calculator: "LandmarksListPaddingCalculator"
//   input_stream: "LANDMARKS:landmarks"
//   output_stream: "TENSORS:tensors"
//   options: {
//     [mediapipe.LandmarksListPaddingCalculatorOptions.ext] {
//       attributes: [X, Y, Z, VISIBILITY, PRESENCE]
//       # flatten: true
//     }
//   }
// }
class LandmarksListPaddingCalculator : public NodeIntf {
 public:
  static constexpr SideInput<int>::Optional kInNumberPoints{"NUMBER_POINTS"};
  static constexpr Input<std::vector<LandmarkList>>::Optional kInLandmarkList{"LANDMARKS"};
  static constexpr Output<std::vector<LandmarkList>>::Optional kOutLandmarkList{"LANDMARKS_PADDING"};
  static constexpr Output<int> kOutHands{"NUMHANDS"};
  MEDIAPIPE_NODE_INTERFACE(LandmarksListPaddingCalculator, kInNumberPoints, kInLandmarkList, kOutLandmarkList, kOutHands);
};

}  // namespace api2
}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_LANDMARKS_LIST_TO_TENSOR_CALCULATOR_H_
