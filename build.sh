#!/bin/bash


echo "开始构建Docker镜像"


docker build \
-t mlops-lite:v2 .


echo "Docker镜像构建完成"
