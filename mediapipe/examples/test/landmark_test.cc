#include <memory>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status_matchers.h"  // NOLINT

namespace mediapipe {

constexpr float kLocationValue = 3;

NormalizedLandmarkList Normalized_GenerateLandmarks(int landmarks_size, int value_multiplier) {
  LOG(INFO) << ">>Call Normalized_GenerateLandmarks...";
  NormalizedLandmarkList landmarks;
  for (int i = 0; i < landmarks_size; ++i) {
    NormalizedLandmark* landmark = landmarks.add_landmark();
    landmark->set_x(value_multiplier * kLocationValue);
    landmark->set_y(value_multiplier * kLocationValue);
    landmark->set_z(value_multiplier * kLocationValue);
  }
  return landmarks;
}

int Normalized_isExistLandmarks(const std::vector<NormalizedLandmarkList>& inputs, const NormalizedLandmarkList& result) {
  LOG(INFO) << ">>Call Normalized_isExistLandmarks...";

  LOG(INFO) << "  >input size " << inputs.size();
  for (int i = 0; i < inputs.size(); ++i) {
    const NormalizedLandmarkList& landmarks_i = inputs[i];
    LOG(INFO) << "    >idx " << i << ", size " << landmarks_i.landmark_size();
  }
  LOG(INFO) << "  >result size " << result.landmark_size();

  return 1;
}

#if 0
TEST(NormalizedLandmarkListCallAsParameterTest, OneTimestamp) {

  LOG(INFO) << ">>Call NormalizedLandmarkListCallAsParameter Calculator.";
  NormalizedLandmarkList input_0 = Normalized_GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  NormalizedLandmarkList input_1 = Normalized_GenerateLandmarks(/*landmarks_size=*/1, /*value_multiplier=*/1);
  NormalizedLandmarkList input_2 = Normalized_GenerateLandmarks(/*landmarks_size=*/2, /*value_multiplier=*/2);
  
  std::vector<NormalizedLandmarkList> inputs = {input_0, input_1, input_2};

  NormalizedLandmarkList result = Normalized_GenerateLandmarks(/*landmarks_size=*/22, /*value_multiplier=*/23);
  Normalized_isExistLandmarks(inputs, result);
}
#endif


//####test LandmarkList##############################################################################

LandmarkList GenerateLandmarks(int landmarks_size, int value_multiplier) {
  LOG(INFO) << ">>Call GenerateLandmarks...";
  LandmarkList landmarks;
  for (int i = 0; i < landmarks_size; ++i) {
    Landmark* landmark = landmarks.add_landmark();
    landmark->set_x(value_multiplier * kLocationValue);
    landmark->set_y(value_multiplier * kLocationValue);
    landmark->set_z(value_multiplier * kLocationValue);
  }
  return landmarks;
}

int isExistLandmarks(const std::vector<LandmarkList>& inputs, const LandmarkList& result) {
  LOG(INFO) << ">>Call isExistLandmarks...";

  LOG(INFO) << "  >input size " << inputs.size();
  for (int i = 0; i < inputs.size(); ++i) {
    const LandmarkList& landmarks_i = inputs[i];
    LOG(INFO) << "    >idx " << i << ", size " << landmarks_i.landmark_size();
  }
  LOG(INFO) << "  >result size " << result.landmark_size();

  return 1;
}

TEST(LandmarkListCallAsParameterTest, OneTimestamp) {

  LOG(INFO) << ">>Call LandmarkListCallAsParameter Calculator.";
  LandmarkList input_0 = GenerateLandmarks(/*landmarks_size=*/3, /*value_multiplier=*/0);
  LandmarkList input_1 = GenerateLandmarks(/*landmarks_size=*/1, /*value_multiplier=*/1);
  LandmarkList input_2 = GenerateLandmarks(/*landmarks_size=*/2, /*value_multiplier=*/2);
  
  std::vector<LandmarkList> inputs = {input_0, input_1, input_2};

  LandmarkList result = GenerateLandmarks(/*landmarks_size=*/22, /*value_multiplier=*/23);
  isExistLandmarks(inputs, result);
}


}  // namespace mediapipe
