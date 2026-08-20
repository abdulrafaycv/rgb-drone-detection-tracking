#!/bin/bash
# Build script for C++ TensorRT Drone Tracker on Jetson Orin Nano

set -e

echo "Building main1.cpp -> main1_cpp..."
g++ -O3 -std=c++17 -o main1_cpp main1.cpp drone_tracker.cpp \
    -I/usr/local/cuda-12.6/targets/aarch64-linux/include \
    -I/usr/include/aarch64-linux-gnu \
    -I/usr/local/include/opencv4 \
    -L/usr/local/cuda-12.6/targets/aarch64-linux/lib \
    -L/usr/lib/aarch64-linux-gnu \
    -L/usr/local/lib \
    -lcudart -lnvinfer $(pkg-config --libs opencv4) \
    -pthread

if [ $? -eq 0 ]; then
    echo "=================================================="
    echo "Build successful! Executable created: ./main1_cpp"
    echo "To run the tracker, execute: ./main1_cpp"
    echo "=================================================="
else
    echo "Build failed!"
    exit 1
fi


