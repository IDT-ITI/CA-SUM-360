# CA-SUM-360
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
For the main train process of the Saliency Detection models, we used the reproduced VR-EyeTracking. For the ATSal method, we first trained the attention model with 85 Salient360! images and 22 Sitzman. The total 107 ERP frames, applying methods like rotate,mirroring and flipping, contains 2140 images, where 1840 used for train and 300 for valdidation. Then, we used the 140 videos from VR-EyeTracking for training and 66 for validation. For the train process of the Expert model, cubemap projection used for Expert_Poles with north and south regions, and for Expert_Equator the front,back,left and right regions. The SST-Sal method trained using the 92 static videos from the total 140 VR-EyeTracking and for validaiton, the 55 static videos from the total 66 validation set
