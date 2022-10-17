#!/bin/sh

# (c) 2022 Christian Hoffmann
# version: 2022-10-13

# This script takes the first *.mp4 file it finds in the directory this script is executed in
# and extracts 3d keypoints from it with. It is a multi-step process that
# - splits videos into left and right parts
# - uses Detectron2 for 2D pose keypoint estimation
# - uses VideoPose3D for 3D extrapolation, both with trajectory and without trajectory
# assumptions:
# - the video file is a synchronized split-screen video which can be separated on the horizontal center

if [ -z "$PS1" ] ; then
    echo "This script must be sourced. Use \"source <script>\" instead."
    exit
fi

###################### 00 - setup ######################

conda activate videopose3d
shopt -s failglob # prevents loops with no found files to run, see https://www.endpointdev.com/blog/2016/12/bash-loop-wildcards-nullglob-failglob/

start_time_full=`date +%s`
path=$PWD # save current folder
allvideofiles=(*.mp4)
videofile=${allvideofiles[0]} #videofile=$PWD/${allvideofiles[0]}
videoname="${videofile//+(*\/|.*)}"

###################### 01 - Split videos (left/right) for separate analysis ######################

start_time_videosplit=`date +%s`
ffmpeg -i $videofile -crf 17 -filter:v "crop=iw/2:ih:0:0" left.mp4
ffmpeg -i $videofile -crf 17 -filter:v "crop=iw/2:ih:iw/2:0" right.mp4
end_time_videosplit=`date +%s`

###################### 02 - Detectron2 analysis ######################

start_time_det=`date +%s`

# create output directories
mkdir "${path}/right"
mkdir "${path}/right/detectron_data"
mkdir "${path}/left"
mkdir "${path}/left/detectron_data"

# move video files to their new positions
mv "${path}/right.mp4" "${path}/right"
mv "${path}/left.mp4" "${path}/left"

# start infering 2D keypoints with Detectron2

cd ~/projects/videopose3d/VideoPose3D/inference

python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir "${path}/right/detectron_data" --image-ext mp4 "${path}/right"
time_det01=`date +%s`
python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir "${path}/left/detectron_data" --image-ext mp4 "${path}/left"
time_det02=`date +%s`

###################### 03 - VideoPose3D analysis ######################

start_time_vp=`date +%s`

mkdir "${path}/right/vp3d_data"
mkdir "${path}/left/vp3d_data"
mkdir "${path}/right/vp3d_visual"
mkdir "${path}/left/vp3d_visual"

cd ../data

python prepare_data_2d_custom.py -i "${path}/right/detectron_data" -o ${videoname}-right
python prepare_data_2d_custom.py -i "${path}/left/detectron_data" -o ${videoname}-left

cd ..

# without trajectory
# coordinates
python run.py -d custom -k ${videoname}-right -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject right.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/right/right.mp4" --viz-export "${path}/right/vp3d_data/right_noTraj.npy" --viz-size 6
time_vp01a=`date +%s`
python run.py -d custom -k ${videoname}-left -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject left.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/left/left.mp4" --viz-export "${path}/left/vp3d_data/left_noTraj.npy" --viz-size 6
time_vp02a=`date +%s`
# visual output
python run.py -d custom -k ${videoname}-right -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject right.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/right/right.mp4" --viz-output "${path}/right/vp3d_visual/right_noTraj.mp4" --viz-size 6
time_vp03a=`date +%s`
python run.py -d custom -k ${videoname}-left -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject left.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/left/left.mp4" --viz-output "${path}/left/vp3d_visual/left_noTraj.mp4" --viz-size 6
time_vp04a=`date +%s`


# with trajectory
# coordinates
python run.py -d custom -k ${videoname}-right -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_243_h36m_detectron_coco_wtraj.bin --render --viz-subject right.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/right/right.mp4" --viz-export "${path}/right/vp3d_data/right_wTraj.npy" --viz-size 6
time_vp01b=`date +%s`
python run.py -d custom -k ${videoname}-left -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_243_h36m_detectron_coco_wtraj.bin --render --viz-subject left.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/left/left.mp4" --viz-export "${path}/left/vp3d_data/left_wTraj.npy" --viz-size 6
time_vp02b=`date +%s`
# visual output
python run.py -d custom -k ${videoname}-right -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_243_h36m_detectron_coco_wtraj.bin --render --viz-subject right.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/right/right.mp4" --viz-output "${path}/right/vp3d_visual/right_wTraj.mp4" --viz-size 6
time_vp03b=`date +%s`
python run.py -d custom -k ${videoname}-left -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_243_h36m_detectron_coco_wtraj.bin --render --viz-subject left.mp4 --viz-action custom --viz-camera 0 --viz-video "${path}/left/left.mp4" --viz-output "${path}/left/vp3d_visual/left_wTraj.mp4" --viz-size 6
time_vp04b=`date +%s`

###################### Final runtime statistics ######################

end_time_full=`date +%s`

echo " " 
echo " " 
echo " "
echo "Video split and encode (right and left) done in `expr $end_time_videosplit - $start_time_videosplit` seconds."
echo ""
echo "1st Detectron2 analysis done in `expr $time_det01 - $start_time_det` seconds."
echo "2nd Detectron2 analysis done in `expr $time_det02 - $time_det01` seconds."
echo "Full time used for Detectron2 analyses: `expr $time_det02 - $start_time_det` seconds."
echo ""
echo "VideoPose3D analysis (right, without trajectory) done in `expr $time_vp01a - $start_time_vp` seconds."
echo "VideoPose3D analysis (left, without trajectory) done in `expr $time_vp02a - $time_vp01a` seconds."
echo "VideoPose3D analysis (right, with trajectory) done in `expr $time_vp01b - $time_vp04a` seconds."
echo "VideoPose3D analysis (left, with trajectory) done in `expr $time_vp02b - $time_vp01b` seconds."
echo "VideoPose3D video output (right, without trajectory) done in `expr $time_vp03a - $time_vp02a` seconds."
echo "VideoPose3D video output (left, without trajectory) done in `expr $time_vp04a - $time_vp03a` seconds."
echo "VideoPose3D video output (right, with trajectory) done in `expr $time_vp03b - $time_vp02b` seconds."
echo "VideoPose3D video output (left, with trajectory) done in `expr $time_vp04b - $time_vp03b` seconds."
echo "Full time used for VideoPose3D analyses: `expr $time_vp02b - $start_time_vp` seconds."
echo ""
echo "Full time used to run script: `expr $end_time_full - $start_time_full` seconds."

cd $path
read -p "Press [Enter] key to exit..."
