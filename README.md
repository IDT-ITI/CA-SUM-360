# CA-SUM-360
## Pytorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360 Videos**
* Written by Evlampios Apostolidis, Ioannis Kontostathis, Vasileios Mezaris
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
│ │ ├── frames/
| | | ├──001/
| | | | ├──0000.png
│ │ ├── saliency/
| | | ├── 001/
| | | | ├──0000.png
│ ├── Salient360!-Sitzaman/
│ │ ├── training/
│ │ | ├── frames/
│ │ | | ├──0000.png
│ │ | ├── saliency/
│ │ | | ├──0000.png
```

## Train models and inference

## Evaluation Results

# Citation

# Licence



