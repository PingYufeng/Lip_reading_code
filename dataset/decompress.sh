#!/bin/bash

DOWNLOAD_DIR=`pwd`/downloaded

DECOMPRESS_DIR=`pwd`/decompressed

VIDEO_PATH=$DOWNLOAD_DIR/video/s#.mpg_vcd.zip
VIDEO_DECOMPRESS_DIR=$DECOMPRESS_DIR/video

ALIGN_PATH=$DOWNLOAD_DIR/align/s#.tar
ALIGN_DECOMPRESS_DIR=$DECOMPRESS_DIR/align

mkdir -p $VIDEO_DECOMPRESS_DIR
mkdir -p $ALIGN_DECOMPRESS_DIR

for i in `seq 1 34`;
do
    if [ $i != 21 ]; then
        video_path=${VIDEO_PATH//#/$i}
        unzip $video_path -d $VIDEO_DECOMPRESS_DIR

        align_path=${ALIGN_PATH//#/$i}
        mkdir $ALIGN_DECOMPRESS_DIR/s$i
        tar -C $ALIGN_DECOMPRESS_DIR/s$i --strip-components=1 -xf $align_path align
    fi
done
