# An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos
## Pytorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos**
* Written by Ioannis Kontostathis, Evlampios Apostolidis, Vasileios Mezaris
* This software can be used for training the deep-learning models for saliency detection and video summarization that are integrated in our method for spatio-temporal summarization of 360-degrees videos, and perform the analysis of a given 360-degrees video in an end-to-end manner. In particular, the 360-degrees video is initially subjected to equirectangular projection (ERP) to form a set of omnidirectional planar (ERP) frames. This set of frames is then analysed by a camera motion detection mechanism (CMDM) which decides on whether the 360-degrees video was captured by a static or a moving camera. Based on this mechanism’s output, the ERP frames are then forwarded to one of the integrated methods for saliency detection (ATSal, SST-Sal), which produce a set of frame-level saliency maps. The ERP frames and the extracted saliency maps are given as input to a component that detects salient events (related to different parts of one or more activities shown in the 360◦ video), computes a saliency score per frame, and produces a 2D video presenting the detected events. The 2D video along with its frames’ saliency scores are fed to a video summarization method (variant of [CA-SUM](https://github.com/e-apostolidis/CA-SUM)) which estimates the importance of each event and forms the video summary.
## Main dependencies
The code for training and evaluating the utilized saliency detection models (ATSal, SST-Sal), was checked and verified on a `Windows 11` PC with an `NVIDIA GeForce GTX 1080Ti` GPU and an `i5-12600K` CPU. Main packages required:
<div align="center">
  <table>
    <tr>
      <th>Python</th>
      <th>PyTorch</th>
      <th>CUDA Version</th>
      <th>cudatoolkit Version</th>
      <th>Numpy</th>
      <th>Opencv</th>
      <th>imageio-ffmpeg</th>
    </tr>
    <tr>
      <td>3.8</td>
      <td>1.7.0</td>
      <td>11.7</td>
      <td>11.0.221</td>
      <td>1.24.3</td>
      <td>4.6.0</td>
      <td>0.4.9</td>
    </tr>
  </table>
</div>

The code for training and evaluating the utilized video summarization model (saliency-aware variant of [CA-SUM](https://github.com/e-apostolidis/CA-SUM)), was checked and verified on an `Ubuntu 20.04.3` PC with an `NVIDIA RTX 2080Ti` GPU and an `i5-11500K` CPU. Main packages required:
<div align="center">
  <table>
    <tr>
      <th>Python</th>
      <th>PyTorch</th>
      <th>CUDA Version</th>
      <th>cuDNN Version</th>
      <th>TensorBoard</th>
      <th>TensorFlow</th>
      <th>Numpy</th>
      <th>H5py</th>
    </tr>
    <tr>
      <td>3.8(.8)</td>
      <td>1.7.1</td>
      <td>11.0</td>
      <td>8005</td>
      <td>2.4.0</td>
      <td>2.4.1</td>
      <td>1.20.2</td>
      <td>2.10.0</td>
    </tr>
  </table>
</div>

## Data
For the train process of the Saliency Detection models, we used the reproduced VR-EyeTracking. For the ATSal method, we first trained the attention model with total 107 ERP frames, applying methods like rotate,mirroring and flipping, contains 2140 images, where 1840 used for train and 300 for valdidation. Then, we used the 140 videos from VR-EyeTracking for training and 66 for validation. For the train process of the Expert model, cubemap projection used for Expert_Poles with north and south regions, and for Expert_Equator the front,back,left and right regions. The SST-Sal method trained using the 92 static videos from the total 140 VR-EyeTracking and for validaiton, the 55 static videos from the total 66 validation set. For the train process of video summarization model we used 100 2D videos that were produced from the 2D Video Production aglorithm and scores in terms of frame level saliency using the methods from Saliency Detection. These videos relate to 46 360 videos of the VR-EyeTracking dataset and 19 from Sports-360 that were captured
using a fixed camera, and 11, 18 and 6 360 videos of the VR-EyeTracking the Sports-360 and the Salient360! datasets, respectively, that were captured by a moving camera.

<div align="justify">

The used data for training the video summarization model are available within the [data](data) folder. The HDF5 file has the following structure:
```Text
/key
    /change_points            2D-array with shape (num_segments, 2), where each row stores the indices of the starting and ending frame of a video segment
    /features                 2D-array with shape (n_steps, 1024), where each row stores the feature vector of the relevant video frame (GoogleNet features)
    /n_frames                 number of video frames
    /n_steps                  number of sampled frames
    /picks                    1D-array with shape (n_steps, 1) with the indices of the sampled frames
    /saliency scores          1D-array with shape (n_steps, 1) with the computed saliency scores for the sampled frames
```
</div>

## Processing Steps

### ERP frame extraction and transformation
To extract the ERP frames from a 360-degrees video use the frames_extractor.py script that is available here and run one of the following commands (we recommend to store the extracted ERP frames in the default path ("data/output_frames"), for easier reference to these frames during the following processing steps):

If the 360-degrees video is in MP4 format, run the following command: 
```
python frames_extractor.py --video_input_type="360" --input_video_path "PATH/path_containing_the_360_videos" --output_folder "data/output_frames"
```
If the 360-degrees video is in ERP format, run the following command:
```
python frames_extractor.py --video_input_type="erp" --input_video_path "PATH/path_containing_the_erp_videos" --output_folder "data/output_frames"  
```
  
To produce the cubemap (CMP) frames and saliency maps that are utilized by the SalEMA expert model of ATSal, use the erp_to_cube.py script that is available here and run the following commands:  
```
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/frames" --equator_save_path "data/Cube_Folder/Equator/frames" --poles_save_path ""data/Cube_Folder/Poles/frames"
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/saliency" --equator_save_path "data/Cube_Folder/Equator/saliency" --poles_save_path ""data/Cube_Folder/Poles/saliency"  
```

To produce the augmentated images of Sitzman and Salient360! images that are utilized by the attention model of ATSal, run the following command:
```
python augmentation.py --path_to_images "data/sitzman-salient360 --path_to_save_augmented_images "data/sitzman-salient360"
```

### Camera Motion Detection Algorithm
To run and use the camera motion detection algorithm, run the following commands:
```
cd CA-SUM-360-main/camera_motion_detection_algorithm
```
```
python cmda.py --frames_folder_path "data\output_frames" --parameter_1 0.5 
```
## Training
For the training process of ATSal model, we first trained the attention model with 2140 images reproduced from 107 ERP images of Salient360! and Sitzman. Then we trained the attention model with 140 VR-EyeTracking videos that is included in the [train_split](data/VR-EyeTracking/train_split.txt) For the fine-tuned train of the Expert models, we used the same videos from VR-EyeTracking but with cube-map projection, applying north and south region to Expert Poles and front,right,back and left to Expert Equator. For the training of SST-Sal, we used 92 static video from VR-EyeTracking named [here](data/Static-VR-EyeTracking), and 55 for validation.
### Dataset
The Salient!360 and Sitzman dataset used for training attention model of ATSal method:
* [Salient360!](https://salient360.ls2n.fr/datasets/training-dataset/) by following the instructions via FTP client.
* [Sitzman](https://drive.google.com/drive/folders/1EJgxC6SzjehWi3bu8PRVHWJrkeZbAiqD) <br>

The reproduced VR-EyeTracking dataset used for training ATSal and SST-Sal:
* [Vr-EyeTracking](https://mtliba.github.io/Reproduced-VR-EyeTracking/)<br>

The Sport-360 dataset used for testing:
* [Sport-360](https://www.terabox.com/sharing/init?surl=nmn4Pb_wmceMmO7QHSiB9Q), with password:4p8t
### Data Structure

```
├── data/
│ ├── VR-EyeTracking/
│ │ ├── frames
│ │ │ ├── 001/
| | | | ├── 0000.png
│ │  ├── saliency/
| | | ├── 001/
| | | | ├── 0000.png
```
To train or evaluate the Saliency_Detection methods, download the datasets and place them in the folder data
### Train-Inference ΑTSal 
ATSal attention model initialization :
* [[intitial (374 MB)]](https://drive.google.com/file/d/1qT4tALLSGmsRfqf_dJ-1nhS_3iT4fFMg/view?usp=sharing)
To train the attention model from scratch, you download the initial weights for the model and place them in the [weights](Saliency_Detection/ATSal/attention/weights) and run the follow command:
```
cd Saliency_Detection/ATSal/attention
```
```
python train.py --gpu "cuda:0" --path_to_frames_folder "data/Salient360-Sitzman/training/frames" --path_to_save_model "Saliency_Detection/ATSal/attention/weights" --batch_size 40
```
The other paramaters are in default mode for the training. To train the model on VR-EyeTracking dataset download the pretrained model weights below: 
ATSal attention model trained on Salient360! and VR-EyeTracking video dataset:
* [[ATSal-Atention-pretrained]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)
save them in the folder [weights](Saliency_Detection/ATSal/attention/weights) and run the follow command in the same folder "attention":
```
python train.py --gpu "cuda:0" --path_to_frames_folder "data/VR-EyeTracking/training/frames" --attention_model "weights/pretrained.pt" --path_to_save_model "Saliency_Detection/ATSal/attention/weights" --batch_size 10 --weight_decay=1e-5
```
To train the expert models download the weight below and place it in this folder [weights](Saliency_Detection/ATSal/expert/weights) and the cube VR-EyeTracking dataset to data folder, 
* [[ATSal-experts-SalEMA30]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)
and run the following command line (example for equator):
```
cd Saliency_Detection/ATSal/expert
```
```
python TrainExpert.py --gpu "cuda:0" --path_to_frames_folder "data/cube-VR-EyeTracking/Equator/training/frames" --clip_size 10 --save_model_path "Saliency_Detection/ATSal/expert/weights"
```
* [[ATSal-experts-Equator (364 MB)]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)
* [[ATSal-experts-Poles (364 MB)]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)

To run an inference of ATSal method to produce saliency maps, you should run and execute the following command (example for ATSal):
```
cd Saliency_Detection/ATSal
```
```
python inference.py --gpu "cuda:0" --path_to_frames_folder "data/VR-EyeTracking/validation/frames" --load_gt "False" --path_to_save_saliency_maps "outputs"
```
### Train-Inference SST-Sal

To train the SST-Sal method run the following commands: 
```
cd Saliency_Detection/SST-Sal
```
```
python train.py python train.py --gpu "cuda:0" --hidden_layers 9 --path_to_training_folder "data/VR-EyeTracking/training/frames" --path_to_validation_folder = "data/VR-EyeTracking/validation/frames" --save_model_path "Saliency_Detection/SST-Sal/weights"
```
SST-Sal model trained on Static-VR-EyeTracking dataset
* [[SST-Sal weights]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8)

To run the inference of the SST-Sal method, you should download the above weights and place them in this [folder](Saliency_Detection/SST-Sal/weights)
then run the commands:
```
cd Saliency_Detection/SST-Sal
```
```
python inference.py --gpu "cuda:0" --path_to_frames_validation_folder "data/VR-EyeTracking/validation/frames" --load_gt "False" --path_to_save_saliency_maps "outputs"
```
### 2D_Video_Production
To create the 2D video, run the following command:
```
python main.py --frames_folder_path "data/VR-EyeTracking/frames" --saliency_maps_path "data/VR-EyeTracking/saliency" --intensity_value 150 --dbscan_distance 1.2 --spatial_distance 100 --fill_loss 100
```
  
# Evaluation
By following the above Data Structure and placing the dataset,weights in the data,weights folder:

To evaluate the ATSal model, run the following command:
```
cd Saliency_Detection/ATSal
```
```
python inference.py --gpu "cuda:0" --path_to_frames_folder "data/VR-EyeTracking/validation/frames" --load_gt "True" --path_to_save_saliency_maps "outputs"
```

To evaluate SST-Sal, run the following command:
```
cd Saliency_Detection/SST-Sal
```
```
python inference.py --gpu "cuda:0" --path_to_frames_validation_folder "data/VR-EyeTracking/validation/frames" --load_gt "True" --path_to_save_saliency_maps "outputs"
```

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgements
This work was supported by the EU's Horizon Europe and Horizon 2020 research and innovation programmes under grant agreements 101070109 TransMIXR and 951911 AI4Media, respectively.

## Citation
If you find our method useful in your work or you use some materials provided in this repo, please cite the following publication where our method and materials were presented: 

````
@inproceedings{kontostathis2024summarization,
    title={An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos},
    author={Kontostathis, Ioannis and Apostolidis, Evlampios and Mezaris, Vasileios},
    year={2024},
    month={Jan.-Feb.},
    booktitle={Proc. 30th Int. Conf. on MultiMedia Modeling (MMM 2024)}
}
````


