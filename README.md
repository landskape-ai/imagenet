# ImageNet
ImageNet Benchmark for different Activation Functions

## Setup the Training Data
Download the ILSVRC2012_img_train.tar and run the bash script `extract_data.sh` present in the `src` folder. This creates a new directory named train and has 1000 subfolders within, which consist of images from the training dataset belonging to the 1000 respective classes.

## Setup the validation data
Download the ILSVRC2012_img_val.tar and run the script `valprep.sh` present in the `src` folder. This creates a new directory named val has 1000 subfolders within, which consist of images from the validation set belonging to their respective classes obtained from the ground_truth.txt present in the ILSVRC DevKit. 

## Runninng Models
1. Running `ShuffleNet_test.py` : To run this code, place code in the same directory as the train and val folders and run `python ShuffleNet_test.py` . 
