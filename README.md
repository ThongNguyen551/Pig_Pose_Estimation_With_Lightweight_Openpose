# Pig_Pose_Estimation_With_Lightweight_Openpose
Coursework project: IMT4392 Deep learning for visual computing - NTNU, Norway

- Project owners: [Milan Kresovic](https://github.com/kresovicmilan), [Thong Nguyen](https://github.com/ThongNguyen551).
- This project is done the transfer-learning based on the implementation [Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) made by [Daniil-Osokin](https://github.com/Daniil-Osokin)
- This project is implemented to work with the pig dataset, which has 6 keypoints in term of the skeleton. 

## Worflow

1. [Training with original Lightweight Openpose](#Training-with-original-Lightweight-Openpose)
2. [Training with pig dataset](#Training-with-pig-dataset)

## Training with original Lightweight Openpose
Follow this [Implementation](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) for all beginning setup, then start training with COCO dataset in order to get the trained weight after 3 stages. This trained weight is used for training with custom dataset later. 

## Training with pig dataset
### Table of Contents

* [Label data](#label-data)
* [Prepare data for training](#prepare-data-for-training)
* [How to set KPTs and PAFs for customised dataset](#How-to-set-KPTs-and-PAFs-for-customised-dataset)
* [Training](#training)
* [Validation](#validation)
* [Video demo](#video-demo)

Our dataset is not public. However, you can send a access request to `oyvind.nordbo@norsvin.no`

#### Label data
For labelling data, we used a fancy tool called [COCO Annotator](https://github.com/jsbroks/coco-annotator). Follow the mentioned link for more detailed instruction. 

#### Prepare data for training
We followed the [TRAIN ON CUSTOM DATASET](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch/blob/master/TRAIN-ON-CUSTOM-DATASET.md) made by [Daniil-Osokin](https://github.com/Daniil-Osokin)

#### How to set KPTs and PAFs for customised dataset
#### Training
```
python train.py --train-images-folder "/path/to/train/datset" --prepared-train-labels "path/to/prepared_train_annotation.pkl" --checkpoint-path "path/to/checkpoint/from/stage3/of/lightweight/openpose" --weights-only "--num-refinement-stages" 3
```
#### Validation
```
python val.py --images-folder "/path/to/validation/datset" --labels "path/to/json/file" --checkpoint-path "path/to/selected/checkpoint"
```
#### Video demo
[![Watch the video](https://studntnu-my.sharepoint.com/:v:/g/personal/thongn_ntnu_no/EY06zEUyqJVMknQYcciWXmsBbCpKUtNcd0Yqgp8iv6fxMg?e=x1xUjW)
