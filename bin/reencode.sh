#!/bin/bash

INFILE=$1
OUTFILE=$2
FRATE=$3 #Set the framerate

# Figure out how to replace file extension of .avi with .mp4

ffmpeg -i $INFILE -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g $FRATE -r $FRATE $OUTFILE
