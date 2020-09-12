#!/bin/bash
# script to download, extract, and prepare ImageNet dataset

# URLs are not public, and must be specified

# ILSVRC2012_img_train.tar (about 138 GB)
TRAIN_URL=""

# ILSVRC2012_img_val.tar (about 6.3 GB)
VALID_URL=""

# optionally download the datasets
if [ -n $TRAIN_URL ]; then wget $TRAIN_URL; fi
if [ -n $VALID_URL ]; then wget $VALID_URL; fi

# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
# after extracting, the expected directory structure is shown below
#
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#

# extract the training data

mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# extract the validation data and move images to subfolders

mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val
tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# check total files after extract

# expected: 1281167
N_TRAIN=find train/ -name "*.JPEG" | wc -l
if [ N_TRAIN -eq 1281167 ]
then
    do echo "Sucessfully extracted training set!"; done
else
    do echo "Something went wrong. Please try again."; done
fi    

# expected: 50000
N_VALID = find val/ -name "*.JPEG" | wc -l
if [ N_VALID -eq 50000 ]
then
    echo "Sucessfully extracted validation set!"
else
    echo "Something went wrong. Please try again."
fi
