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
#include <iostream>
#include <fstream>
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

constexpr char kInputSidePacketStream[] = "number_points";
constexpr char kInputStream[] = "input_image";
constexpr char kOutputStream[] = "output_image";
constexpr char kOutputLandmarkStream[] = "multi_hand_world_landmarks";
constexpr char kOutputLandmarkPaddingStream[] = "multi_hand_world_landmarks_padding";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_image_path, "", "Full path of image file.");
ABSL_FLAG(std::string, number_points, "", "Number of extend padding points to add.");

absl::Status WriteOffFile(const std::vector<mediapipe::LandmarkList> collection)
{
    LOG(INFO) << "Collection size " << collection.size();
    for (const auto& landmarkList : collection) {
        LOG(INFO) << " >landmark size = " << landmarkList.landmark_size();
    }
}

#if 1
absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    //LOG(INFO) << "Get calculator graph config contents: \n" << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    //LOG(INFO) << "Initialize the input image.";
    if(absl::GetFlag(FLAGS_input_image_path).empty())
    {
        LOG(ERROR) << "Missing image.";
        return absl::OkStatus();
    }
    
    int number_points = 0;
    if(!absl::GetFlag(FLAGS_number_points).empty())
    {
        number_points = atoi(absl::GetFlag(FLAGS_number_points).c_str());
    }

    LOG(INFO) << "Initialize the calculator graph. Padding number points " << number_points;

    std::map<std::string, mediapipe::Packet> input_side_packets;
    if(number_points > 0)
    {
        input_side_packets[kInputSidePacketStream] = mediapipe::MakePacket<int>(number_points);
    }

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));


    //cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark, graph.AddOutputStreamPoller(kOutputLandmarkStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_padding, graph.AddOutputStreamPoller(kOutputLandmarkPaddingStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start processing frames.";
    bool grab_frames = true;
    cv::Mat camera_frame_raw = cv::imread(absl::GetFlag(FLAGS_input_image_path));
    LOG(INFO) << "Load image file " << absl::GetFlag(FLAGS_input_image_path);
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Ignore empty frames";
      return absl::OkStatus();
    }
    
    /*
    cv::imshow(kWindowName, camera_frame_raw);
    cv::waitKey(0);
    */
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
#if 0
    if(poller.Next(&packet))
    {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        
        
        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        // Press any key to exit.
        cv::waitKey(0);
    }
#endif

    if(poller_landmark.Next(&packet))
    {
        const auto& collection = packet.Get<std::vector<mediapipe::LandmarkList>>();
        LOG(INFO) << kOutputLandmarkStream << " Collection size " << collection.size();
        
        std::string fname = "/share/hand.off";
        std::ofstream outfile;
        outfile.open(fname);
        outfile << "OFF" << std::endl;
        int total_landmark = 0;
        for (const auto& landmarkList : collection) {
            total_landmark += landmarkList.landmark_size();
        }
        outfile << total_landmark << " 0 0" << std::endl;
        for (const auto& landmarkList : collection) {
            LOG(INFO) << " >" << kOutputLandmarkStream << " landmark size = " << landmarkList.landmark_size();
            
            int cnt = 0;
            for (int i = 0; i < landmarkList.landmark_size(); ++i)
            {
                cnt++;
                const auto& landmark = landmarkList.landmark(i);
                //LOG(INFO) << "  ==>" << cnt << " (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
                outfile << landmark.x() << " " << landmark.y() << " " << landmark.z() << std::endl;
            }
        }
        outfile.close();
        LOG(INFO) << ">>Write off file " << fname;
    }    

    if(poller_landmark_padding.Next(&packet))
    {
        const auto& collection = packet.Get<std::vector<mediapipe::LandmarkList>>();
        LOG(INFO) << kOutputLandmarkPaddingStream << " Collection size " << collection.size();
        
        std::string fname = "/share/hand_padding.off";
        std::ofstream outfile;
        outfile.open(fname);
        outfile << "OFF" << std::endl;
        int total_landmark = 0;
        for (const auto& landmarkList : collection) {
            total_landmark += landmarkList.landmark_size();
        }
        outfile << total_landmark << " 0 0" << std::endl;
        for (const auto& landmarkList : collection) {
            LOG(INFO) << " >" << kOutputLandmarkPaddingStream << " landmark size = " << landmarkList.landmark_size();
            
            int cnt = 0;
            for (int i = 0; i < landmarkList.landmark_size(); ++i)
            {
                cnt++;
                const auto& landmark = landmarkList.landmark(i);
                //LOG(INFO) << "  ==>" << cnt << " (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
                outfile << landmark.x() << " " << landmark.y() << " " << landmark.z() << std::endl;
            }
        }
        outfile.close();
        LOG(INFO) << ">>Write off file " << fname;
    }    
    
    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

#else

absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    //LOG(INFO) << "Get calculator graph config contents: \n" << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    //LOG(INFO) << "Initialize the input image.";
    if(absl::GetFlag(FLAGS_input_image_path).empty())
    {
        LOG(ERROR) << "Missing image.";
        return absl::OkStatus();
    }
    
    int number_points = 0;
    if(!absl::GetFlag(FLAGS_number_points).empty())
    {
        number_points = atoi(absl::GetFlag(FLAGS_number_points).c_str());
    }

    LOG(INFO) << "Initialize the calculator graph. Padding number points " << number_points;

    std::map<std::string, mediapipe::Packet> input_side_packets;
    if(number_points > 0)
    {
        input_side_packets[kInputSidePacketStream] = mediapipe::MakePacket<int>(number_points);
    }

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));


    //cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    LOG(INFO) << "Start running the calculator graph.";
    //ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
    //ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark, graph.AddOutputStreamPoller(kOutputLandmarkStream));
        
    auto dataOutputStream = graph.AddOutputStreamPoller(kOutputLandmarkStream);
    assert(dataOutputStream.ok());
    std::unique_ptr<mediapipe::OutputStreamPoller> poller_data = std::make_unique<mediapipe::OutputStreamPoller>(std::move(dataOutputStream.value()));
    assert(poller_data != nullptr);

    auto videoOutputStream = graph.AddOutputStreamPoller(kOutputStream);
    assert(videoOutputStream.ok());
    std::unique_ptr<mediapipe::OutputStreamPoller> poller_video = std::make_unique<mediapipe::OutputStreamPoller>(std::move(videoOutputStream.value()));
    assert(poller_video != nullptr);

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start processing frames.";
    bool grab_frames = true;
    cv::Mat camera_frame_raw = cv::imread(absl::GetFlag(FLAGS_input_image_path));
    LOG(INFO) << "Load image file " << absl::GetFlag(FLAGS_input_image_path);
    if (camera_frame_raw.empty()) {
      LOG(INFO) << "Ignore empty frames";
      return absl::OkStatus();
    }
    
    /*
    cv::imshow(kWindowName, camera_frame_raw);
    cv::waitKey(0);
    */
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
    if(poller_video->Next(&packet))
    {
        auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        
#if 0
        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
        cv::imshow(kWindowName, output_frame_mat);
        // Press any key to exit.
        cv::waitKey(0);
#endif
    }

    if(poller_data->QueueSize() > 0)
    {
        if(poller_data->Next(&packet))
        {
            //LOG(INFO) << "Found landmark packet.";
#if 0
            const auto& collection = packet.Get<std::vector<mediapipe::LandmarkList>>();
            LOG(INFO) << "Collection size " << collection.size();
            for (const auto& landmarkList : collection) {
                LOG(INFO) << " >landmark size = " << landmarkList.landmark_size();
            }
#else
            const auto& collection = packet.Get<std::vector<mediapipe::LandmarkList>>();
            WriteOffFile(collection);
#endif
        }
    }    
    
    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    if(poller_data)
    {
        //delete poller_data;
        poller_data = nullptr;
    }

    if(poller_video)
    {
        //delete poller_video;
        poller_video = nullptr;
    }

    return graph.WaitUntilDone();
}
#endif


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
