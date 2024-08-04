#!/bin/bash

# Define the URL and target directory
URL="https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Training_Input.zip"
TARGET_DIR="data/"

# Create the target directory
mkdir -p $TARGET_DIR

# Download the zip file using curl
curl -o /tmp/ISIC_2024_Training_Input.zip $URL

curl -o /data/metadata.csv https://isic-challenge-data.s3.amazonaws.com/2024/ISIC_2024_Training_GroundTruth.csv

# Extract the contents of the zip file to the target directory
unzip -q /tmp/ISIC_2024_Training_Input.zip -d $TARGET_DIR

# Clean up the downloaded zip file
rm /tmp/ISIC_2024_Training_Input.zip

# Notify the user of completion
echo "Download and extraction complete. Files are located in $TARGET_DIR"
