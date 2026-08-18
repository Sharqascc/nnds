#!/bin/bash
# Download pre-trained models for the pipeline
# Place your Google Drive file IDs or direct URLs below

echo "Creating data/models directory..."
mkdir -p data/models

echo "Downloading UVH model (uvh26.pt)..."
# Replace YOUR_UVH_FILE_ID with the actual Google Drive ID
# gdown --id YOUR_UVH_FILE_ID -O data/models/uvh26.pt

echo "Downloading YOLO model (yolo11n.pt)..."
# gdown --id YOUR_YOLO_FILE_ID -O data/models/yolo11n.pt

echo "Downloading RT-DETR model (rtdetr-l.pt)..."
# gdown --id YOUR_RTDETR_FILE_ID -O data/models/rtdetr-l.pt

echo "Done! Models saved to data/models/"
