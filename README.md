# An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos
## PyTorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos**
* Written by Ioannis Kontostathis, Evlampios Apostolidis, Vasileios Mezaris
* This software can be used for training the deep-learning models for saliency detection and video summarization that are integrated in our method for spatio-temporal summarization of 360-degrees videos, and perform the analysis of a given 360-degrees video in an end-to-end manner. In particular, the 360-degrees video is initially subjected to equirectangular projection (ERP) to form a set of omnidirectional planar (ERP) frames. This set of frames is then analysed by a camera motion detection mechanism (CMDM) which decides on whether the 360-degrees video was captured by a static or a moving camera. Based on this mechanism’s output, the ERP frames are then forwarded to one of the integrated methods for saliency detection ([ATSal](https://github.com/mtliba/ATSal/tree/master), [SST-Sal](https://github.com/edurnebernal/SST-Sal)), which produce a set of frame-level saliency maps. The ERP frames and the extracted saliency maps are given as input to a component that detects salient events (related to different parts of one or more activities shown in the 360◦ video), computes a saliency score per frame, and produces a 2D video presenting the detected events. The 2D video along with its frames’ saliency scores are fed to a video summarization method (variant of [CA-SUM](https://github.com/e-apostolidis/CA-SUM)) which estimates the importance of each event and forms the video summary.
## Main dependencies
The code for training and evaluating the utilized saliency detection models ([ATSal](https://github.com/mtliba/ATSal/tree/master), [SST-Sal](https://github.com/edurnebernal/SST-Sal)), was checked and verified on a `Windows 11` PC with an `NVIDIA GeForce GTX 1080Ti` GPU and an `i5-12600K` CPU. Main packages required:
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

To train and evaluate the saliency detection models, we used the following datasets:

* The Salient360! dataset, that is publicly-available [here](https://salient360.ls2n.fr/datasets/training-dataset/) (follow the instructions to download the dataset using an FTP client)
* The Sitzman dataset that, is publicly-available [here](https://drive.google.com/drive/folders/1EJgxC6SzjehWi3bu8PRVHWJrkeZbAiqD)
* A re-produced version of the VR-EyeTraking dataset, that is publicly-available [here](https://mtliba.github.io/Reproduced-VR-EyeTracking/)
* The Sport-360 dataset (only for testing), that is publicly-available [here](https://www.terabox.com/sharing/init?surl=nmn4Pb_wmceMmO7QHSiB9Q) (password:4p8t)

To train the video summarization model, we used the data stored in [CA-SUM-360.h5](.....). The HDF5 file has the following structure:
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

## Processing components

### ERP frame extraction and transformation
To extract the ERP frames from a 360-degrees video use the [frames_extractor.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/scripts/frames_extractor.py) script and run one of the following commands (we recommend to store the extracted ERP frames in the default path ("data/output_frames"), for easier reference to these frames during the following processing steps):

If the 360-degrees video is in MP4 format, run the following command: 
```
python frames_extractor.py --video_input_type="360" --input_video_path "PATH/path_containing_the_360_videos" --output_folder "data/output_frames"
```
If the 360-degrees video is in ERP format, run the following command:
```
python frames_extractor.py --video_input_type="erp" --input_video_path "PATH/path_containing_the_erp_videos" --output_folder "data/output_frames"  
```
  
To produce the cubemap (CMP) frames and saliency maps that are utilized by the SalEMA expert model of ATSal, use the [erp_to_cube.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/scripts/erp_to_cube.py) script and run the following commands:  
```
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/cmp_frames/equator/training/frames" --equator_save_path "data/VR-EyeTracking/cmp_frames/poles/training/frames" --poles_save_path ""data/Cube_Folder/Poles/frames"
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/cmp_frames/equator/training/saliency" --equator_save_path "data/VR-EyeTracking/cmp_frames/poles/training/saliency" --poles_save_path ""data/Cube_Folder/Poles/saliency"  
```

### Camera motion detection
To run the camera motion detection mechanism (CMDM) on the extracted ERP frames, use the [cmdm.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/camera_motion_detection_algorithm/cmdm.py) script and run the following command:
```
python cmdm.py --frames_folder_path "data\output_frames" --parameter_1 0.5 
```

### Saliency detection

To extract saliency maps for the ERP frames based on the ATSal method, download the pre-trained models [[ATSal-Equator.pt]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8) and [[ATSal-Poles.pt]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8) and [[ATSal-Attention.pt]](https://drive.google.com/file/d/1Ke-Ad7lwME6kZdaW8_PUCU7MNklaeJRA/view?usp=sharing), store them in the "Saliency_Detection/ATSal/attention/weights" directory, use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/ATSal) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames ".../data/<dataset-name>/ERP_frames" --load_gt "False" --path_to_extracted_saliency_maps "...data/<dataset-name>/extracted_saliency_maps"
```

To extract saliency maps for the ERP frames based on the SST-Sal method, download the pre-trained model [[SST-Sal weights]](https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8), store it in the "Saliency_Detection/SST-Sal/weights" directory, use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/SST-Sal) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames ".../data/<dataset-name>/ERP_frames" --load_gt "False" --path_to_extracted_saliency_maps "...data/<dataset-name>/extracted_saliency_maps"
```

### Salient event detection and 2D video production

To detect the salient events in the 360-degrees video, and formulate the conventional 2D video that contains these events, use the [main.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/2D_Video_Production/main.py) script and run the following command:
```
python main.py --path_to_ERP_frames ".../data/<dataset-name>/ERP_frames" --path_to_extracted_saliency_maps "...data/<dataset-name>/extracted_saliency_maps" --intensity_value 150 --dbscan_distance 1.2 --spatial_distance 100 --fill_loss 100
```
The produced MPEG-4 video file and the computed saliency scores for its frames, will be stored in ...


### Video summarization

To be added


## Training
The attention model of ATSal was initially trained using a dataset of 2140 ERP images, created after applying common data augmentation operations (rotation, cropping, etc.) on 107 ERP images of the Salient360! and Sitzman datasets; 1840 of these images were used for training and the remaining 300 for validation (the corresponding ERP frames are listed [here](link)). 

To re-create this dataset: i) download the Salient360! dataset from [here](https://salient360.ls2n.fr/datasets/training-dataset/) (follow the instruction to download the dataset using an FTP client), ii) download the Sitzman dataset from [here](https://drive.google.com/drive/folders/1EJgxC6SzjehWi3bu8PRVHWJrkeZbAiqD), and iii) run the [augmentation.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/scripts/augmentation.py) script to perform the data augmentation process, as follows:

```
python augmentation.py .... to be defined
```

To train the attention model using this dataset, use the [train.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/ATSal/attention/train.py) script and run the following command:

```
python train.py --gpu "cuda:0" --path_to_frames_folder "data/Salient360-Sitzman/training/frames" --model_storage_path "Saliency_Detection/ATSal/attention/weights" --batch_size 40
```

Following, the attention model of ATSal was trained using the 206 videos from the VR-EyeTracking dataset from [here](https://mtliba.github.io/Reproduced-VR-EyeTracking/); 140 of them were used for training (listed [here](data/VR-EyeTracking/train_split.txt)) and the remaining 66 of them for validation.

Finally, pre-trained models of the SalEMA Expert of ATSal (available [here]([https://drive.google.com/drive/folders/1fTMrH00alyZ_hP7CaYenkzIkFevRRVz8](https://github.com/Linardos/SalEMA))) were trained using the CMP frames of the same 206 videos from VR-EyeTracking (following the same split of data into training and validation set). Frames presenting the north and south regions of the ERP frames (stored in .../path/) are used to train the SalEMA Expert Poles model, while frames presenting the front, back, right and left regions of the ERP frames (stored in .../path/) are used to train the SalEMA Expert Equator model. 

To train SST-Sal, we used only the static videos from the VR-EyeTracking dataset. In total, we used 92 videos for training (listed [here](data/Static-VR-EyeTracking)) and 55 videos (listed [here](data/Static-VR-EyeTracking)) for validation.

For the train process of video summarization model we used 100 2D videos that were produced from the 2D Video Production aglorithm and scores in terms of frame level saliency using the methods from Saliency Detection. These videos relate to 46 360 videos of the VR-EyeTracking dataset and 19 from Sports-360 that were captured using a fixed camera, and 11, 18 and 6 360 videos of the VR-EyeTracking the Sports-360 and the Salient360! datasets, respectively, that were captured by a moving camera.

### Train-Inference ΑTSal 
ATSal attention model initialization :
* [[initial]](https://drive.google.com/file/d/1qT4tALLSGmsRfqf_dJ-1nhS_3iT4fFMg/view?usp=sharing)
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
python TrainExpert.py --gpu "cuda:0" --path_to_training_cmp_frames "data/VR-EyeTracking/cmp_frames/equator/training/frames" --path_to_validation_cmp_frames "data/VR-EyeTracking/cmp_frames/equator/training/frames" --clip_size 10 --save_model_path "Saliency_Detection/ATSal/expert/weights"
```

### Train-Inference SST-Sal

To train the SST-Sal method run the following commands: 
```
cd Saliency_Detection/SST-Sal
```
```
python train.py python train.py --gpu "cuda:0" --hidden_layers 9 --path_to_training_folder "data/VR-EyeTracking/training/frames" --path_to_validation_folder = "data/VR-EyeTracking/validation/frames" --save_model_path "Saliency_Detection/SST-Sal/weights"
```

  
# Evaluation
By following the above Data Structure and placing the dataset,weights in the data,weights folder:

To evaluate the ATSal model, run the following command:
```
cd Saliency_Detection/ATSal
```
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --load_gt "True" 
```

To evaluate SST-Sal, run the following command:
```
cd Saliency_Detection/SST-Sal
```
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --load_gt "True" 
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


