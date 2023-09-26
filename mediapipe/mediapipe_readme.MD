[资料]
https://sites.google.com/view/andrewsworkshop/mediapipe%E6%95%99%E5%AD%B8%E8%92%90%E9%9B%86
https://www.cnblogs.com/Iflyinsky/p/14697882.html
https://blog.csdn.net/luozhichengaichenlei/article/details/117262688
*****************https://www.796t.com/article.php?id=66759
https://www.796t.com/article.php?id=67245
https://www.796t.com/article.php?id=128818

*******https://www.cnblogs.com/zhongzhaoxie/category/1843398.html


[自定义3D眼镜]
https://github.com/google/mediapipe/issues/1155
https://github.com/google/filament

#0.[build test extention]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/test_extention:test_extention_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/test_extention/test_extention_cpu --calculator_graph_config_file=mediapipe/graphs/test_extention/test_extention_cpu.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4

[subimpl]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/sub_impl:subimpl_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/sub_impl/subimpl_cpu --calculator_graph_config_file=mediapipe/graphs/sub_impl/subimpl_cpu.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4


#1.[build hello_world]

bazel run --sandbox_debug --verbose_failures --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world

bazel run --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/hello_world:hello_world

export GLOG_logtostderr=1
bazel run --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/hello_world:hello_world
bazel run --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hello_world:hello_world

#2.[hair_segmentation]

##compile
bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/hair_segmentation:hair_segmentation_cpu
bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hair_segmentation:hair_segmentation_cpu

##run
GLOG_logtostderr=1
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4

#########################
##compile
bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/hair_segmentation:hair_segmentation_cpu_image

##run
GLOG_logtostderr=1
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu_image --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live.pbtxt
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu_image --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4


bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hair_segmentation:hair_segmentation_cpu_color
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu_color --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live_color.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu_color --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live_color.pbtxt --input_video_path=/share/image/face5.jpg


#3.[hand_tracking]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

>>GPU
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt

[face_detection]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt



bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu_novideo
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu_novideo --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live_novideo.pbtxt --input_video_path=/home/postgres/workspace/project/dlib/example/images

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu_lib
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu_lib --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live_novideo.pbtxt --input_video_path=/home/postgres/workspace/project/dlib/example/images




bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu_image
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu_image --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/home/postgres/workspace/project/dlib/example/person/me31.jpg
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu_image --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/home/postgres/workspace/project/dlib/example/images/face_detect.jpg


####[face detect lib]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/librect:face_detection
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/libsubface:subface
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt

#****************************
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_detection:face_detection_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_detection/face_detection_cpu --calculator_graph_config_file=mediapipe/graphs/face_detection/face_detection_desktop_live.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4

[face_mesh]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/face_mesh:face_mesh_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt


bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_mesh:face_mesh_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_mesh_cpu --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt  --input_video_path=/share/selfie_segmentation_web.mp4

[face effect]

bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/face_effect:face_effect_gpu_deps
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hair_segmentation/hair_segmentation_cpu_color --calculator_graph_config_file=mediapipe/graphs/hair_segmentation/hair_segmentation_desktop_live_color.pbtxt --input_video_path=/share/selfie_segmentation_web.mp4


[holistic_tracking]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/holistic_tracking:holistic_tracking_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/holistic_tracking/holistic_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt

[iris_tracking]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/iris_tracking:iris_tracking_cpu

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu.pbtxt
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_depth_cpu.pbtxt
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/iris_tracking/iris_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/iris_tracking/iris_tracking_cpu_video_input.pbtxt

[object_detection]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/object_detection:object_detection_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt

*************
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt --input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live.pbtxt --input_image_path=/share/image2.jpg


bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu_16lite
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu_16lite --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live_16lite.pbtxt --input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu_mobile
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu_mobile --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live_mobile.pbtxt --input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4

*************
#https://github.com/google/mediapipe/issues/2009
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/object_detection:object_detection_cpu
GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_detection/object_detection_cpu --calculator_graph_config_file=mediapipe/graphs/object_detection/object_detection_desktop_live_custom.pbtxt --input_video_path=mediapipe/examples/desktop/object_detection/test_video.mp4

[object_tracking]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/object_tracking:object_tracking_cpu

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/object_tracking/object_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/object_tracking/object_tracking_desktop_live.pbtxt

[pose_tracking]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH=/usr/bin/python2 mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/pose_tracking/pose_tracking_cpu --calculator_graph_config_file=mediapipe/graphs/pose_tracking/pose_tracking_cpu.pbtxt


[target_demo]
bazel build -c opt --sandbox_debug --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/target_demo:target_demo_cpu

GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/target_demo/target_demo_cpu --calculator_graph_config_file=mediapipe/graphs/target_demo/target_demo_desktop.pbtxt --input_image_path=/share/image/face5.jpg

[aar apk]
[aar face effect]
[3d models]
#https://grabcad.com/library?page=3&per_page=100&time=all_time&sort=recent&query=glasses
#https://www.cgtrader.com/free-3d-models?keywords=glasses
#epay.kuangzy@126.com/kuangzy123
#https://zhuanlan.zhihu.com/p/40680702

3D对齐
https://threejs.org/editor/

3d模型拓扑轻量化
https://www.laozicloud.com/

bazel build -c opt --linkopt="-s" \
--host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
--fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_faceeffect:hair_faceeffect_aar

bazel build -c opt --linkopt="-s" \
--host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
--fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_objectdetect:aar_objectdetect



bazel build -c opt mediapipe/graphs/face_effect:face_effect_gpu_binary_graph

bazel build -c opt mediapipe/graphs/face_effect/data:axis
bazel build -c opt mediapipe/graphs/face_effect/data:glasses

bazel build -c opt mediapipe/graphs/face_effect/data:sunglasses
bazel build -c opt mediapipe/graphs/face_effect/data:glasses4
bazel build -c opt mediapipe/graphs/face_effect/data:hair
bazel build -c opt mediapipe/graphs/face_effect/data:hair4

###子图不需要单独编译
#bazel build -c opt mediapipe/graphs/face_effect/subgraphs:face_landmarks_smoothing
#bazel build -c opt mediapipe/graphs/face_effect/subgraphs:single_face_geometry_from_detection_gpu
#bazel build -c opt mediapipe/graphs/face_effect/subgraphs:single_face_geometry_from_landmarks_gpu
#bazel build -c opt mediapipe/graphs/face_effect/subgraphs:multi_render_face_effect

bazel build -c opt mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_detection
bazel build -c opt mediapipe/modules/face_geometry/data:geometry_pipeline_metadata_landmarks

bazel build -c opt mediapipe/modules/face_geometry/protos:face_geometry_proto

./android-arm64-v8a-opt/bin/mediapipe/framework/_javac/calculator_proto/libcalculator_proto-lite_temp/com/google/mediapipe/proto/CalculatorProto.java
./android-arm64-v8a-opt/bin/mediapipe/framework/_javac/calculator_profile_proto/libcalculator_profile_proto-lite_temp/com/google/mediapipe/proto/CalculatorProfileProto.java
./android-arm64-v8a-opt/bin/mediapipe/modules/face_geometry/protos/_javac/mesh_3d_proto/libmesh_3d_proto-lite_temp/com/google/mediapipe/modules/facegeometry/Mesh3dProto.java
./android-arm64-v8a-opt/bin/mediapipe/modules/face_geometry/protos/_javac/face_geometry_proto/libface_geometry_proto-lite_temp/com/google/mediapipe/modules/facegeometry/FaceGeometryProto.java
[postgres@fprint bazel-out]$ 


[apk face effect]
bazel build -c opt --config=android_arm64 mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceeffect:faceeffect
adb install bazel-bin/mediapipe/examples/android/src/java/com/google/mediapipe/apps/faceeffect/faceeffect.apk


[aar object detect]
bazel build -c opt --linkopt="-s" \
--host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
--fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_objectdetect:aar_objectdetect


bazel build -c opt --linkopt="-s" \
--host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
--fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/examples/android/src/java/com/google/mediapipe/apps/aar_objectdetect:aar_objectdetect


bazel build -c opt mediapipe/graphs/object_detection:mobile_gpu_binary_graph

bazel build -c opt mediapipe/graphs/object_detection:mobile_gpu_custom_binary_graph

bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_gpu
bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_cpu

bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_gpu_landmark
bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_landmark

bazel build -c opt mediapipe/modules/holistic_landmark:holistic_tracking_gpu_landmark_image
bazel build -c opt mediapipe/modules/holistic_landmark:holistic_tracking_cpu_landmark_image

#bazel build -c opt mediapipe/modules/holistic_landmark:holistic_landmark_tracking_gpu_image
#bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/modules/holistic_landmark:holistic_landmark_tracking_cpu_image

bazel build -c opt mediapipe/java/com/google/mediapipe/solutions/hands/hands


bazel build -c opt mediapipe/graphs/palm_tracking:palm_detection_gpu_image


###编译solution-core
bazel build -c opt --linkopt="-s" \
--host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
--fat_apk_cpu=arm64-v8a,armeabi-v7a //mediapipe/java/com/google/mediapipe/solutioncore:solution_core

bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_gpu_image
bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_gpu_image2
bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_gpu_image_example
bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_cpu_image_example
bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_gpu_image_binary
bazel build -c opt mediapipe/graphs/hand_tracking:hand_landmark_tracking_cpu_image_binary
                                                      
bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_landmark_image
bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_gpu_landmark_image

bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_gpu_landmark_image
bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_cpu_landmark_image

bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_gpu_binary_graph
bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_cpu_binary_graph

bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_gpu_binary_graph2
bazel build -c opt mediapipe/graphs/pose_tracking:pose_tracking_cpu_binary_graph2

bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_gpu_image
bazel build -c opt mediapipe/graphs/holistic_tracking:holistic_tracking_gpu_video

############################################
[测试例子编译/运行]

[image]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/image:image_cropping_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/image/image_cropping_calculator_test

[core]

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/core:flow_limiter_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/core/flow_limiter_calculator_test

[tensor]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/tensor:inference_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/tensor/inference_calculator_test



bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/tensor:tensors_to_binary_classification_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/tensor/tensors_to_binary_classification_calculator_test




[util]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/util:hand_gesture_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/util/hand_gesture_calculator_test



bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/util:detections_to_rects_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/util/detections_to_rects_calculator_test

[tflite]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/tflite:tflite_tensors_to_classification_calculator_test

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/tflite/tflite_tensors_to_classification_calculator_test


##demo
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/calculators/demo:my_string_process_calculator

GLOG_logtostderr=1 bazel-bin/mediapipe/calculators/demo/my_string_process_calculator


[framework]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/framework:calculator_graph_test

GLOG_logtostderr=1 bazel-bin/mediapipe/framework/calculator_graph_test

[framework api2]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/framework/api2:subgraph_test

GLOG_logtostderr=1 bazel-bin/mediapipe/framework/api2/subgraph_test

[framework]
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/framework/tool:switch_container_test

GLOG_logtostderr=1 bazel-bin/mediapipe/framework/tool/switch_container_test


[calculator 添加opencv库支持]
--calculator.cc

        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"


[code]
#1. ImageFrame -> cv::Mat ->format
  cv::Mat input_mat;
  mediapipe::ImageFormat::Format format;

  const auto& input = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
  input_mat = formats::MatView(&input);
  format = input.Format();


#---------------
    const auto& image_frame = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    const int height = image_frame.Height();
    const int width = image_frame.Width();
    const int channels = image_frame.NumberOfChannels();
    const int channels_preserved = std::min(channels, max_num_channels_);
    const mediapipe::ImageFormat::Format format = image_frame.Format();



#【tensorflow】浅谈什么是张量tensor(https://blog.csdn.net/qq_31821675/article/details/79188449)

#YOLOv5 Android(完结) - (https://blog.csdn.net/rush9838465/article/details/113647107)
#YOLOV5部署安卓 - https://zhuanlan.zhihu.com/p/401711864
#YOLOv5 NCNN Implementation - https://github.com/sunnyden/YOLOV5_NCNN_Android



#######################################################
tflite_tensors_to_classification_calculator_test.cc

landmarks_to_tensor_calculator_test.cc
tensors_to_classification_calculator_test.cc


