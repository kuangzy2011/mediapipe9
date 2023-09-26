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
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <iomanip>

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

constexpr char kWindowName[] = "MediaPipe";
constexpr char kInputStream[] = "input_image";
constexpr char kOutputStream[] = "output_image";
constexpr char kOutputLandmarkStream[] = "multi_hand_world_landmarks";
constexpr char kOutputLandmarkPaddingStream[] = "multi_hand_world_landmarks_padding";
constexpr char kOutputLandmarkPresenceStream[] = "landmark_presence";
constexpr char kInputSidePacketStream[] = "number_points";

ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_image_path, "", "Parent path of image to pick up landmark.");
ABSL_FLAG(std::string, number_points, "", "Number of extra landmark points to padding between current connection.");
ABSL_FLAG(std::string, output_off_path, "", "Save the OFF file to this path.");
ABSL_FLAG(std::string, sleep_time, "", "Process image file interval, default is 100000 microsecond, unit is microsecond.");
ABSL_FLAG(std::string, filter_landmaks, "", "If filter_landmarks is not 0, filter all landmarks is not same with specified landmarks.");
ABSL_FLAG(std::string, single_class, "", "Only marks the single class.");
ABSL_FLAG(std::string, show_render_image_of_inconsist_landmarks, "", "If landmarks is inconsist with filter_landmarks, show rendered image.");

mediapipe::CalculatorGraph graph;
std::map<std::string, mediapipe::OutputStreamPoller *> pollerList;
//mediapipe::OutputStreamPoller *ptr_poller_landmark_presence = nullptr;
int sleep_time = 100000;
int filter_landmaks = 0;
int show_image = 0;
int single_class = 0;

template <class... Args>
int string_format(std::string& format, Args&&... args)
{
    auto size_buf = std::snprintf(nullptr, 0, format.c_str(), std::forward<Args>(args)...) + 1;
    std::unique_ptr<char[]> buf = std::make_unique<char[]>(size_buf);
    if (!buf)
    {
        return 0;
    }

    std::snprintf(buf.get(), size_buf, format.c_str(), std::forward<Args>(args)...);
    format = std::string(buf.get(), buf.get() + size_buf - 1);
    return format.length();
}

mediapipe::OutputStreamPoller* getPoller(std::string name) {
    mediapipe::OutputStreamPoller* ptr = nullptr;
        
    auto poller = pollerList.find(name);
    if(poller != pollerList.end()) {
        ptr = poller->second;
    }
    
    return ptr;
} 

int processLandmark(const std::vector<mediapipe::LandmarkList> &connection, std::string output_file) {
    //LOG(INFO) << "[I] - call processLandmark. size " << connection.size();
    
    std::string content;
    int total_landmark = 0;    
    int idx = 0;
    
    for(auto& landmarkList: connection) {
        //LOG(INFO) << "[I] - idx " << idx << ", size " << landmarkList.landmark_size();
        total_landmark += landmarkList.landmark_size();
    }
    
    //filter landmarks
    if(filter_landmaks > 0 && total_landmark != filter_landmaks) {
        LOG(INFO) << "[W] - processLandmark - filter landmarks " << total_landmark;
        return 1;
    }
    
    std::ofstream outfile;
    outfile.open(output_file);
	//outfile.setf(std::ios_base::scientific, std::ios_base::floatfield);
    outfile.setf(std::ios_base::fixed);
	outfile << std::setprecision(6); //force to output 6 float
    outfile << "OFF" << std::endl;
    outfile << total_landmark << " 0 0" << std::endl;

    for(auto& landmarkList: connection) {
        //LOG(INFO) << "[I] - idx " << idx << ", size " << landmarkList.landmark_size();
        for (int i = 0; i < landmarkList.landmark_size(); ++i) {
            auto& landmark = landmarkList.landmark(i);
            //LOG(INFO) << "    >i " << i << ", (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
            outfile << landmark.x();
            outfile << " ";
            outfile << landmark.y();
            outfile << " ";
            outfile << landmark.z() << std::endl;
        }
    }
    
    outfile.close();

    LOG(INFO) << "[I] - Write file " << output_file;

    return 0;
}

absl::Status RunProcesserClassOff(const std::string input_file, const std::string output_file) {
    cv::Mat camera_frame_raw = cv::imread(input_file);
    if (camera_frame_raw.empty()) {
        return absl::OkStatus();
    }
    
    LOG(INFO) << "[D] - RunProcesserClassOff - input_file " << input_file;
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
          mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);
    
    // Send image packet into the graph.
    size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));
    
    mediapipe::Packet packet;
    cv::Mat output_frame_mat;
    int show = 0;
    // Get the graph result packet, or stop if that fails.
    if(show_image > 0) {
        auto poller_image = getPoller(kOutputStream);
        if(poller_image != nullptr) {
            if(poller_image->Next(&packet)) {
                auto& output_frame = packet.Get<mediapipe::ImageFrame>();
                output_frame_mat = mediapipe::formats::MatView(&output_frame);
                cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            }
        }
    }

    // Get the graph result packet, or stop if that fails.
    auto poller_presence = getPoller(kOutputLandmarkPresenceStream);
    if(poller_presence != nullptr) {
        if(poller_presence->Next(&packet)) {
            auto has_landmark = packet.Get<bool>();
            if(has_landmark) {
                auto poller_landmark = getPoller(kOutputLandmarkPaddingStream);
                if(poller_landmark != nullptr) {
                    if(poller_landmark->Next(&packet))
                    {
                        auto& output_landmark = packet.Get<std::vector<mediapipe::LandmarkList>>();
                        
                        show = processLandmark(output_landmark, output_file);
                    }
                }
            }
            else {
                LOG(INFO) << "[W] - RunProcesserClassOff - No landmarks";
            }
        }
    }
    
    if(show_image > 0 && show > 0) {
        cv::imshow(kWindowName, output_frame_mat);
        cv::waitKey(0);
    }
    
    return absl::OkStatus();
}

absl::Status RunProcesserClass(const std::string input_path, const std::string output_path) {
    struct stat s;
    struct dirent *filename;
    DIR *dir = NULL;

    LOG(INFO) << "[D] - RunProcesserClass input path " << input_path;
    
    lstat(input_path.c_str(), &s);
    if(!S_ISDIR(s.st_mode)){
        LOG(ERROR) << "[E] - input path " << input_path.c_str() << " is not a valid directory!";
        return mediapipe::StatusBuilder(absl::StatusCode::kUnknown, MEDIAPIPE_LOC).SetPrepend();
    }
    
    std::string opath = "%s/%s";
    string_format(opath, output_path.c_str(), basename(input_path.c_str()));
    //LOG(INFO) << "[D] - RunProcesserClass opath: " << opath;

    std::string command = "mkdir -p %s";
    string_format(command, opath.c_str());
    //LOG(INFO) << "[D] - RunProcesserClass execute: " << command;
    system(command.c_str());


    dir = opendir(input_path.c_str());
    if(NULL == dir){
        LOG(ERROR) << "[E] - RunProcesserClass Cannot open input path " << input_path.c_str();
        return mediapipe::StatusBuilder(absl::StatusCode::kUnknown, MEDIAPIPE_LOC).SetPrepend();
    }

    std::string ifile;
    std::string ofile;
    int filecnt = 1;
    while ((filename = readdir(dir)) != NULL)
    {
        if(strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
            continue;
        }

        ifile = "%s/%s";
        string_format(ifile, input_path.c_str(), filename->d_name);
        ofile = "%s/%s_%d.off";
        string_format(ofile, opath.c_str(), filename->d_name, filecnt);
        lstat(ifile.c_str(), &s);
        if(!S_ISDIR(s.st_mode)) {
            RunProcesserClassOff(ifile, ofile);
            filecnt++;
        }
    }
    
    closedir(dir);

    return absl::OkStatus();
}

absl::Status RunProcesser(const std::string input_path, const std::string output_path) {
    struct stat s;
    struct dirent *filename;
    DIR *dir = NULL;

    LOG(INFO) << "[D] - RunProcesser input path " << input_path << ", output path " << output_path;
    
    lstat(input_path.c_str(), &s);
    if(!S_ISDIR(s.st_mode)){
        LOG(ERROR) << "[E] - input path " << input_path.c_str() << " is not a valid directory!";
        return mediapipe::StatusBuilder(absl::StatusCode::kUnknown, MEDIAPIPE_LOC).SetPrepend();
    }


    dir = opendir(input_path.c_str());
    if(NULL == dir){
        LOG(ERROR) << "[E] - RunProcesser Cannot open input path " << input_path.c_str();
        return mediapipe::StatusBuilder(absl::StatusCode::kUnknown, MEDIAPIPE_LOC).SetPrepend();
    }
    
    char command[256] = {0};
    sprintf(command, "mkdir -p %s", output_path.c_str());
    system(command);
    
    while ((filename = readdir(dir)) != NULL)
    {
        if(strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
            continue;
        }

        memset(command, 0, sizeof(command));
        sprintf(command, "%s/%s", input_path.c_str(), filename->d_name);
        lstat(output_path.c_str(), &s);
        if(S_ISDIR(s.st_mode)) {
            RunProcesserClass(std::string(command), output_path);
        }
    }
    
    closedir(dir);
    
    return absl::OkStatus();
}

absl::Status RunMPPGraph() {
    int number_points = 0;
    std::map<std::string, mediapipe::Packet> input_side_packets;
    
    RET_CHECK(!absl::GetFlag(FLAGS_calculator_graph_config_file).empty()) << ": Missing graph config file.";
    
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &calculator_graph_config_contents));
    //LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);
    
    RET_CHECK(!absl::GetFlag(FLAGS_input_image_path).empty()) << ": Missing input path.";
    RET_CHECK(!absl::GetFlag(FLAGS_output_off_path).empty()) << ": Missing output path.";
       
    if(!absl::GetFlag(FLAGS_sleep_time).empty()) {
        sleep_time = atoi(absl::GetFlag(FLAGS_sleep_time).c_str());
    }    

    if(!absl::GetFlag(FLAGS_filter_landmaks).empty()) {
        filter_landmaks = atoi(absl::GetFlag(FLAGS_filter_landmaks).c_str());
    }    

    if(!absl::GetFlag(FLAGS_single_class).empty()) {
        single_class = atoi(absl::GetFlag(FLAGS_single_class).c_str());
    }    

    //show_render_image_of_inconsist_landmarks
    if(!absl::GetFlag(FLAGS_show_render_image_of_inconsist_landmarks).empty()) {
        show_image = atoi(absl::GetFlag(FLAGS_show_render_image_of_inconsist_landmarks).c_str());
    }    
    
    if(!absl::GetFlag(FLAGS_number_points).empty()) {
        number_points = atoi(absl::GetFlag(FLAGS_number_points).c_str());
        input_side_packets[kInputSidePacketStream] = mediapipe::MakePacket<int>(number_points);
    }
    
    if(show_image > 0) {
        cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    }
    
    LOG(INFO) << "[I] - Initialize the calculator graph.";
    //mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
    
    
    LOG(INFO) << "[I] - Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_image, graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark, graph.AddOutputStreamPoller(kOutputLandmarkPaddingStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmark_presence, graph.AddOutputStreamPoller(kOutputLandmarkPresenceStream));
    MP_RETURN_IF_ERROR(graph.StartRun({}));
    
    pollerList[kOutputStream] = &poller_image;
    pollerList[kOutputLandmarkPaddingStream] = &poller_landmark;
    pollerList[kOutputLandmarkPresenceStream] = &poller_landmark_presence;
    
    if(single_class) {
        RunProcesserClass(absl::GetFlag(FLAGS_input_image_path), absl::GetFlag(FLAGS_output_off_path));
    }
    else {
        RunProcesser(absl::GetFlag(FLAGS_input_image_path), absl::GetFlag(FLAGS_output_off_path));
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
