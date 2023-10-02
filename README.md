# CA-SUM-360
## Pytorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360 Videos**
* Written by Ioannis Kontostathis, Evlampios Apostolidis, Vasileios Mezaris
* The input 360 video is initially subjected to equirectangular projection (ERP).This set of
frames is given to the decision mechanism, analyze them and makes a decision on whether the
360 video has been captured by a static or a moving camera.
So, based on the output of the decision mechanism the ERP frames are
subsequently forwarded to one of the integrated methods for saliency detection,
which produce a set of frame-level saliency maps( ATSal for moving camera and SST-Sal for static). The ERP frames and the extracted
saliency maps are then given as input to a component that converts the
360 video into a 2D video containing the detected salient events. Finally, the
produced 2D video is feeded to the CA-SUM-360 that estimates about the importance of each frame of the video and forms the video
summary.
## Main dependencies
Saliency Detection models, checked and verified on an `Windows 11` PC with an `NVIDIA GeForce GTX 1080Ti` GPU and an `i5-12600K` CPU. Main packages required:
|`Python` | `PyTorch` | `CUDA Version` | `cudatoolkit Version` | `Numpy` | `Opencv` |
:---:|:---:|:---:|:---:|:---:|:---:|
3.8 | 1.7.0 | 11.7 | 11.0.221 | 1.24.3 | 4.6.0 
## Data
For the train process of the Saliency Detection models, we used the reproduced VR-EyeTracking. For the ATSal method, we first trained the attention model with total 107 ERP frames, applying methods like rotate,mirroring and flipping, contains 2140 images, where 1840 used for train and 300 for valdidation. Then, we used the 140 videos from VR-EyeTracking for training and 66 for validation. For the train process of the Expert model, cubemap projection used for Expert_Poles with north and south regions, and for Expert_Equator the front,back,left and right regions. The SST-Sal method trained using the 92 static videos from the total 140 VR-EyeTracking and for validaiton, the 55 static videos from the total 66 validation set. For the train process of video summarization model we used 100 2D videos that were produced from the 2D Video Production aglorithm and scores in terms of frame level
saliency using the methods from Saliency Detection. These videos relate to 46 360
videos of the VR-EyeTracking dataset and 19 from Sports-360 that were captured
using a fixed camera, and 11, 18 and 6 360 videos of the VR-EyeTracking
the Sports-360 and the Salient360! datasets, respectively, that were captured by
a moving camera

## Training
For the training process of ATSal model, we first trained the attention model with 2140 images reproduced from 107 ERP images of Salient360! and Sitzman. Then we trained the attention model with 140 VR-EyeTracking videos that is included in the [train_split](data/VR-EyeTracking/train_split.txt) For the fine-tuned train of the Expert models, we used the same videos from VR-EyeTracking but with cube-map projection, applying north and south region to Expert Poles and front,right,back and left to Expert Equator. For the training of SST-Sal, we used 92 static video from VR-EyeTracking named here, and 55 for validation.

### Data Structure

```
├── data/
│ ├── VR-EyeTracking/
│ │ ├── training/
│ │ │ ├── frames/
| | | | ├── 001/
| | | | | ├── 0000.png
│ │ | ├── saliency/
| | | | ├── 127/
| | | | | ├── 0000.png
│ │ ├── validation/
│ │ │ ├── frames/
| | | | ├── 001/
| | | | | ├── 0000.png
│ │ | ├── saliency/
| | | | ├── 127/
| | | | | ├── 0000.png
```

## Train models and inference
ATSal attention model initialization :
* [[intitial (374 MB)]](https://drive.google.com/file/d/1qT4tALLSGmsRfqf_dJ-1nhS_3iT4fFMg/view?usp=sharing)

ATSal attention model trained on Salient360! and VR-EyeTracking video dataset:
* [[ATSal-attention (374 MB)]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)

ATSal expert models fine-tuned on VR-EyeTracking video dataset:
* [[ATSal-experts-Poles (364 MB)]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)
* [[ATSal-experts-Equator (364 MB)]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)

SST-Sal model trained on Static-VR-EyeTracking dataset
* [[SST-Sal]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)

To run an inference of Saliency Detection method to produce saliency maps, you should first download the weights of the models to the weights folder in each of ATSal, SST-Sal you want to run and execute the following command (example for ATSal):
```
cd Saliency_Detection/ATSal
```
```
python inference.py --path_to_frames_folder "CA-SUM-360/data/VR-EyeTracking/validation/frames" --load_gt "False" --path_to_save_saliency_maps "CA-SUM-360/outputs_folder"
```
To train the SST-Sal method, follow the following commands
```
cd Saliency_Detection/SST-Sal
```
```
python train.py --gpu "cuda:0" --hidden_layers 9 --path_to_training_folder "CA-SUM-360/data/VR-EyeTracking/training/frames" --path_to_validation_folder = "CA-SUM-360/data/VR-EyeTracking/validation/frames" --save_model_path "CA-SUM-360/Saliency_Detection/SST-Sal/weights"
``` 
## Evaluation Results

# Citation

# Licence



