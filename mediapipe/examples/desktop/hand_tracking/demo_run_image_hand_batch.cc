//#[CTOUCH]
// Copyright 2019 The MediaPipe Authors.
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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_image";
constexpr char kOutputStream[] = "output_image";
constexpr char kOutputLandmarkStream[] = "multi_hand_world_landmarks";
constexpr char kOutputLandmarkPaddingStream[] = "multi_hand_world_landmarks_padding";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kInputSidePacketStream[] = "number_points";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_image_path, "", "Full path of image to pick up landmark.");
ABSL_FLAG(std::string, number_points, "", "Number of extra landmark points to padding between current connection.");
ABSL_FLAG(std::string, showing, "", "Show the rendered image.");
ABSL_FLAG(std::string, output_off_path, "", "Save the OFF file to this path.");

absl::Status processLandmark(const std::vector<mediapipe::LandmarkList> &connection) {
    LOG(INFO) << "[I] - call processLandmark. size " << connection.size();
    
    int idx = 0;
    for(auto& landmarkList: connection) {
        LOG(INFO) << "[I] - idx " << idx << ", size " << landmarkList.landmark_size();
        for (int i = 0; i < landmarkList.landmark_size(); ++i) {
            auto& landmark = landmarkList.landmark(i);
            LOG(INFO) << "    >i " << i << ", (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
        }
    }
    
    

    return absl::OkStatus();
}

absl::Status RunMPPGraph() {
    int number_points = 0;
    int is_show = 0;
    std::map<std::string, mediapipe::Packet> input_side_packets;
    
    RET_CHECK(!absl::GetFlag(FLAGS_calculator_graph_config_file).empty()) << ": Missing graph config file.";
    
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    //LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    
    RET_CHECK(!absl::GetFlag(FLAGS_input_image_path).empty()) << ": Missing image file.";
    
    if(!absl::GetFlag(FLAGS_showing).empty()) {
        is_show = 1;
    }
    
    if(is_show) {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    }
    else {
        LOG(INFO) << "[W] - Ignore to show image.";
    }
    
    if(!absl::GetFlag(FLAGS_number_points).empty()) {
        number_points = atoi(absl::GetFlag(FLAGS_number_points).c_str());
        input_side_packets[kInputSidePacketStream] = mediapipe::MakePacket<int>(number_points);
    }    
    
    LOG(INFO) << "[I] - Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
    
    
    LOG(INFO) << "[I] - Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_padding, graph.AddOutputStreamPoller(kOutputLandmarkPaddingStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    
    LOG(INFO) << "[I] - Start processing frames.";
    LOG(INFO) << "[I] - Load image file " << absl::GetFlag(FLAGS_input_image_path);
    cv::Mat camera_frame_raw = cv::imread(absl::GetFlag(FLAGS_input_image_path));
    if (camera_frame_raw.empty()) {
        LOG(INFO) << "[E] - Ignore empty frames";
        return absl::OkStatus();
    }
    
    if(is_show) {
        cv::imshow(kWindowName, camera_frame_raw);
        cv::waitKey(0);
    }
    
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    //cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
     
    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
          mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);
    
    // Send image packet into the graph.
    size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));
    
    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if(poller.Next(&packet))
    {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
    
        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
          
        if(is_show) {
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            cv::imshow(kWindowName, output_frame_mat);
            // Press any key to exit.
            cv::waitKey(0);
        }
    }

    if(poller_padding.Next(&packet))
    {
        auto& output_landmark = packet.Get<std::vector<mediapipe::LandmarkList>>();
        
        processLandmark(output_landmark);
    }
    
    LOG(INFO) << "[I] - Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
        
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "[E] - Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    }
    else {
        LOG(INFO) << "Success!";
    }
    
    return EXIT_SUCCESS;
}
