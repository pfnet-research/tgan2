#!/bin/bash

if [ ! -d datasets ]; then
    mkdir datasets
fi

cd datasets

if [ ! -d ucf101 ]; then
    mkdir ucf101
fi

cd ucf101

wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e UCF101.rar

wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm -rf UCF101TrainTestSplits-RecognitionTask.zip
