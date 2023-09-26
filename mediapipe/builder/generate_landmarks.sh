#!/bin/bash

MEDIAPIPE_HOME=/home/postgres/workspace/download/mediapipe/mediapipe-0.9.1
#input_image_path=/home/postgres/workspace/project/data/data/right20230327_resize5/train
#output_off_path=/home/postgres/workspace/project/data/off/right20230327_resize5_30/train
single_class=1
number_points=50
filter_landmaks=$((21*(50+1)))
sleep_time=200000

function log() {
    echo $1
}

function processClass() {
    log "Process class ${1}"
    GLOG_logtostderr=1 ${MEDIAPIPE_HOME}/bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_landmark_to_off --calculator_graph_config_file=${MEDIAPIPE_HOME}/mediapipe/graphs/hand_tracking/hand_tracking_desktop_image.pbtxt --input_image_path=$1 --output_off_path=$2 --single_class=$single_class --filter_landmaks=$filter_landmaks --number_points=$number_points --sleep_time=$sleep_time
}

function process() {
    log "Process path ${1}"
    
    for name in `ls $1`; do
        path=${1}/${name}
        processClass $path $2
    done
}

function train() {
    #process "/home/postgres/workspace/project/data/data/right20230327_resize5/train" "/home/postgres/workspace/project/data/off/right20230327_resize5_30/train"
    #process "/home/postgres/workspace/project/data/data/right20230327/train" "/home/postgres/workspace/project/data/off/right20230327_30/train"
    process "/home/postgres/workspace/project/data/data/right20230327/train" "/home/postgres/workspace/project/data/off/right20230327_50/train"
}

function valid() {
    #process "/home/postgres/workspace/project/data/data/right20230327_resize5/valid" "/home/postgres/workspace/project/data/off/right20230327_resize5_30/valid"
    #process "/home/postgres/workspace/project/data/data/right20230327/valid" "/home/postgres/workspace/project/data/off/right20230327_30/valid"
    process "/home/postgres/workspace/project/data/data/right20230327/valid" "/home/postgres/workspace/project/data/off/right20230327_50/valid"
}

if [ $1 == "train" ]; then
    train
fi

if [ $1 == "valid" ]; then
    valid
fi
