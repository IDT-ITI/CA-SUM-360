# CA-SUM-360: 360-degrees video summarization
## Pytorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos**
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
<div align="center">
  <table>
    <tr>
      <th>Python</th>
      <th>PyTorch</th>
      <th>CUDA Version</th>
      <th>cudatoolkit Version</th>
      <th>Numpy</th>
      <th>Opencv</th>
    </tr>
    <tr>
      <td>3.8</td>
      <td>1.7.0</td>
      <td>11.7</td>
      <td>11.0.221</td>
      <td>1.24.3</td>
      <td>4.6.0</td>
    </tr>
  </table>
</div>

## Data
For the train process of the Saliency Detection models, we used the reproduced VR-EyeTracking. For the ATSal method, we first trained the attention model with total 107 ERP frames, applying methods like rotate,mirroring and flipping, contains 2140 images, where 1840 used for train and 300 for valdidation. Then, we used the 140 videos from VR-EyeTracking for training and 66 for validation. For the train process of the Expert model, cubemap projection used for Expert_Poles with north and south regions, and for Expert_Equator the front,back,left and right regions. The SST-Sal method trained using the 92 static videos from the total 140 VR-EyeTracking and for validaiton, the 55 static videos from the total 66 validation set. For the train process of video summarization model we used 100 2D videos that were produced from the 2D Video Production aglorithm and scores in terms of frame level
saliency using the methods from Saliency Detection. These videos relate to 46 360
videos of the VR-EyeTracking dataset and 19 from Sports-360 that were captured
using a fixed camera, and 11, 18 and 6 360 videos of the VR-EyeTracking
the Sports-360 and the Salient360! datasets, respectively, that were captured by
a moving camera

## Training
For the training process of ATSal model, we first trained the attention model with 2140 images reproduced from 107 ERP images of Salient360! and Sitzman. Then we trained the attention model with 140 VR-EyeTracking videos that is included in the [train_split](data/VR-EyeTracking/train_split.txt) For the fine-tuned train of the Expert models, we used the same videos from VR-EyeTracking but with cube-map projection, applying north and south region to Expert Poles and front,right,back and left to Expert Equator. For the training of SST-Sal, we used 92 static video from VR-EyeTracking named [here](data/Static-VR-EyeTracking), and 55 for validation.

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


