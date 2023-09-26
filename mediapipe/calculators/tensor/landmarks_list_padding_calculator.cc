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

#include "mediapipe/calculators/tensor/landmarks_list_padding_calculator.h"

#include <memory>

#include "mediapipe/calculators/tensor/landmarks_list_padding_calculator.pb.h"
#include "mediapipe/framework/api2/node.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {
namespace api2 {

namespace {

float GetAttribute(const Landmark& landmark, const LandmarksListPaddingCalculatorOptions::Attribute& attribute) {
  switch (attribute) {
    case LandmarksListPaddingCalculatorOptions::X:
      return landmark.x();
    case LandmarksListPaddingCalculatorOptions::Y:
      return landmark.y();
    case LandmarksListPaddingCalculatorOptions::Z:
      return landmark.z();
    case LandmarksListPaddingCalculatorOptions::VISIBILITY:
      return landmark.visibility();
    case LandmarksListPaddingCalculatorOptions::PRESENCE:
      return landmark.presence();
  }
}

mediapipe::LandmarkList *GenerateExtendPoints(const Landmark& startPoint, const Landmark& endPoint, int number_points, mediapipe::LandmarkList *landmarkList) {
  if(landmarkList == NULL) {
    return NULL;
  }
  
  for(int i = 1; i <= number_points; i++) {
    float k = (float)i / (number_points - i + 1);
    
    Landmark p1; 
    p1.set_x(endPoint.x() * k);
    p1.set_y(endPoint.y() * k);
    p1.set_z(endPoint.z() * k);
    
    Landmark p2; 
    p2.set_x(startPoint.x() + p1.x());
    p2.set_y(startPoint.y() + p1.y());
    p2.set_z(startPoint.z() + p1.z());
    
    Landmark* p3 = landmarkList->add_landmark();
    p3->set_x(p2.x() / (k + 1));
    p3->set_y(p2.y() / (k + 1));
    p3->set_z(p2.z() / (k + 1));    
  }
  
  return landmarkList;
}


}  // namespace

class LandmarksListPaddingCalculatorImpl: public NodeImpl<LandmarksListPaddingCalculator> {
 public:
/*
  static absl::Status GetContract(CalculatorContract* cc) {
    LOG(INFO) << ">>LandmarksListPaddingCalculator GetContract ";
    if(cc->InputSidePackets().HasTag("NUMBER_POINTS"))
    {
        cc->InputSidePackets().Tag("NUMBER_POINTS").Set<int>().Optional();
    }

    return absl::OkStatus();
  }
*/  
  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options<LandmarksListPaddingCalculatorOptions>();
    RET_CHECK(options_.attributes_size() > 0) << "At least one attribute must be specified";
    
    struct Connection connection;
    connection.start = 0;
    connection.end = 1;
    connections.push_back(connection);
    connection.start = 1;
    connection.end = 2;
    connections.push_back(connection);
    connection.start = 2;
    connection.end = 3;
    connections.push_back(connection);
    connection.start = 3;
    connection.end = 4;
    connections.push_back(connection);
    
    connection.start = 0;
    connection.end = 5;
    connections.push_back(connection);
    connection.start = 5;
    connection.end = 6;
    connections.push_back(connection);
    connection.start = 6;
    connection.end = 7;
    connections.push_back(connection);
    connection.start = 7;
    connection.end = 8;
    connections.push_back(connection);

    connection.start = 5;
    connection.end = 9;
    connections.push_back(connection);
    connection.start = 9;
    connection.end = 10;
    connections.push_back(connection);
    connection.start = 10;
    connection.end = 11;
    connections.push_back(connection);
    connection.start = 11;
    connection.end = 12;
    connections.push_back(connection);

    connection.start = 9;
    connection.end = 13;
    connections.push_back(connection);
    connection.start = 13;
    connection.end = 14;
    connections.push_back(connection);
    connection.start = 14;
    connection.end = 15;
    connections.push_back(connection);
    connection.start = 15;
    connection.end = 16;
    connections.push_back(connection);

    connection.start = 13;
    connection.end = 17;
    connections.push_back(connection);

    connection.start = 0;
    connection.end = 17;
    connections.push_back(connection);
    connection.start = 17;
    connection.end = 18;
    connections.push_back(connection);
    connection.start = 18;
    connection.end = 19;
    connections.push_back(connection);
    connection.start = 19;
    connection.end = 20;
    connections.push_back(connection);

    /*
    LOG(INFO) << "Connections size " << connections.size();
    for (const auto& conn : connections) {
      LOG(INFO) << "conn (" << conn.start << ", " << conn.end << ")";
    }
    */

    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    if (kInLandmarkList(cc).IsEmpty()) {
      return absl::OkStatus();
    }
    
    // Get input vector landmarks
    const auto collection = *kInLandmarkList(cc);
    if(collection.size() == 0)
    {
      return absl::OkStatus();
    }
    
    int c_size = collection.size(); //number hands, single hand is 1
    int n_landmarks = collection.front().landmark_size(); //number points of one hand landmark, 21
    RET_CHECK(c_size <= 2) << "Max number hands is 2";
    
    //LOG(INFO) << ">>LandmarksListPaddingCalculator n_landmarks " << n_landmarks << ", c_size " << c_size;

    //Validate landmark
    for (const auto& landmarkList : collection) {
        RET_CHECK(landmarkList.landmark_size() == n_landmarks) << "landmark size confused, n_landmarks = " << n_landmarks << ", landmark_size = " << landmarkList.landmark_size();
    }
    
    
    
    
    
    //Enhance to generate extend landmark points
    if (options_.intensify()) {
      int number_points = 0;
      
      if(cc->InputSidePackets().HasTag("NUMBER_POINTS")) {
        number_points = cc->InputSidePackets().Tag("NUMBER_POINTS").Get<int>();
        //LOG(INFO) << ">>LandmarksListPaddingCalculator side packet number points: " << number_points;
      }
      else {
          if(options_.has_number_points()) {
            number_points = options_.number_points();
            //LOG(INFO) << ">>LandmarksListPaddingCalculator config number points: " << number_points;
          }
      }
      
      
      //LOG(INFO) << ">>LandmarksListPaddingCalculator number points: " << number_points;
      if(number_points > 0) {
        for (const auto& landmarkList : collection) {
          //Landmark* landmark = landmarkList.add_landmark();
          for(const auto& conn: connections) {
            GenerateExtendPoints(landmarkList.landmark(conn.start), landmarkList.landmark(conn.end), number_points, (mediapipe::LandmarkList *)&landmarkList);
          }          
        }
        
        n_landmarks = collection.front().landmark_size();
      }
    }
    
    
    
    
    /*
    // Determine tensor shape.
    const int n_attributes = options_.attributes_size(); //size of {X, Y, Z}
    auto tensor_shape = options_.flatten() ? Tensor::Shape{1, c_size * n_landmarks * n_attributes} : Tensor::Shape{1, c_size * n_landmarks, n_attributes};

    // Create empty tesnor.
    Tensor tensor(Tensor::ElementType::kFloat32, tensor_shape);
    auto* buffer = tensor.GetCpuWriteView().buffer<float>();

    // Fill tensor with landmark attributes.
    int k = 0;
    for (const auto& landmarkList : collection) {
      LOG(INFO) << "landmarkList size " << landmarkList.landmark_size() << ", n_landmarks " << n_landmarks;
      for (int i = 0; i < n_landmarks; ++i) {
        for (int j = 0; j < n_attributes; ++j) {
          buffer[k * n_landmarks * n_attributes + i * n_attributes + j] = GetAttribute(landmarkList.landmark(i), options_.attributes(j));
        }
      }
      k++;
    }

    // Return vector with a single tensor.
    auto result = std::vector<Tensor>();
    result.push_back(std::move(tensor));
    kOutTensors(cc).Send(std::move(result));
    */
    //std::vector<mediapipe::LandmarkList> aa;
    if(cc->Outputs().HasTag("LANDMARKS_PADDING")) {
        kOutLandmarkList(cc).Send(std::move(collection));
        //kOutLandmarkList(cc).Send(std::move(aa));
    }
    

    //LOG(INFO) << ">>>>>>>>>>>>>>>>>>LandmarksListPaddingCalculator num hands: " << c_size;
    if(cc->Outputs().HasTag("NUMHANDS")) {
      if(c_size == 1) {
        kOutHands(cc).Send(0);        
      }
      else {
        kOutHands(cc).Send(1);
      }
    }

    return absl::OkStatus();
  }

 private:
  LandmarksListPaddingCalculatorOptions options_;
  struct Connection {
    int start;
    int end;
  };
  std::vector<struct Connection> connections;
  
};
MEDIAPIPE_NODE_IMPLEMENTATION(LandmarksListPaddingCalculatorImpl);

}  // namespace api2
}  // namespace mediapipe
