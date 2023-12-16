# An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos
## PyTorch implementation of CA-SUM-360
* from **An Integrated System for Spatio-Temporal Summarization of 360-degrees Videos**
* Written by Ioannis Kontostathis, Evlampios Apostolidis, Vasileios Mezaris
* This software can be used for training the deep-learning models for saliency detection and video summarization that are integrated in our method for spatio-temporal summarization of 360-degrees videos, and perform the analysis of a given 360-degrees video in an end-to-end manner. In particular, the 360-degrees video is initially subjected to equirectangular projection (ERP) to form a set of omnidirectional planar (ERP) frames. This set of frames is then analysed by a camera motion detection mechanism (CMDM) which decides on whether the 360-degrees video was captured by a static or a moving camera. Based on this mechanism’s output, the ERP frames are then forwarded to one of the integrated methods for saliency detection ([ATSal](https://github.com/mtliba/ATSal/tree/master), [SST-Sal](https://github.com/edurnebernal/SST-Sal)), which produce a set of frame-level saliency maps. The ERP frames and the extracted saliency maps are given as input to a component that detects salient events (related to different parts of one or more activities shown in the 360-degrees video), computes a saliency score per frame, and produces a 2D video presenting the detected events. The 2D video along with its frames’ saliency scores are fed to a video summarization method (variant of [CA-SUM](https://github.com/e-apostolidis/CA-SUM)) which estimates the importance of each event and forms the video summary.
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
* The Sport-360 dataset (only for testing), that is publicly-available [here](https://github.com/vhchuong1997/Saliency-prediction-for-360-degree-video)

To train the video summarization model, we used the created [360VideoSumm](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/Video-Summarization/360VideoSumm.h5). The associated HDF5 file has the following structure:
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

## Pre-processing step

### ERP frame extraction and transformation
To extract the ERP frames from a 360-degrees video use the [frames_extractor.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Preprocessing_scripts/frames_extractor.py) script and run one of the following commands (we recommend to store the extracted ERP frames in the default path ("data/output_frames"), for easier reference to these frames during the following processing steps):

If the 360-degrees video is in MP4 format, run the following command: 
```
python frames_extractor.py --video_input_type "360" --input_video_path "PATH/path_containing_the_360_videos" --output_folder "data/erp_videos"
```
If the 360-degrees video is in ERP format, run the following command:
```
python frames_extractor.py --video_input_type "erp" --input_video_path "PATH/path_containing_the_erp_videos" --output_folder "data/erp_videos"  
```
  
To produce the cubemap (CMP) frames and saliency maps that are utilized by the SalEMA expert model of ATSal, use the [erp_to_cube.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Preprocessing_scripts/erp_to_cube.py) script and run the following commands:  
```
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/erp_frames/frames" --equator_save_path "data/VR-EyeTracking/cmp_frames/equator/training/frames" --poles_save_path "data/VR-EyeTracking/cmp_frames/poles/training/frames"
python erp_to_cube.py --path_to_erp_video_frames "data/VR-EyeTracking/erp_frames/saliency" --equator_save_path "data/VR-EyeTracking/cmp_frames/equator/training/saliency" --poles_save_path data/VR-EyeTracking/cmp_frames/poles/training/saliency"  
```

## Processing steps

### Camera motion detection
To run the camera motion detection mechanism (CMDM) on the extracted ERP frames, use the [cmdm.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Camera_Motion_Detection/cmdm.py) script and run the following command:
```
python cmdm.py --path_to_frames_folder "data\output_frames" --parameter_1 0.5 
```

### Saliency detection

#### ATSal method

The attention model of ATSal was initially trained using a dataset of 2140 ERP images, created after applying common data augmentation operations (rotation, cropping, flipping etc.) on 107 ERP images of the Salient360! and Sitzman datasets; 1840 of these images were used for training and the remaining 300 for validation. This dataset is available [here](https://drive.google.com/file/d/1BTxs6E3Wnk-nlVgu-lGzYgr1kmNse9T9/view?usp=sharing). 

To train the attention model using the aforementioned dataset, download the initial instance of this model (called "ATSal-Attention-Initial") from [[here]](https://drive.google.com/file/d/15-pl9drbAZSYnL-5b-C63K7-JUE8nAsJ/view?usp=sharing), place it in the [weights](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/ATSal/weights) directory, use the [train.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/ATSal/attention/train.py) script and run the following command:
```
python train.py --gpu "cuda:0" --path_to_ERP_frames "data/Salient360-Sitzman/training/frames" --dataset "Salient360!-Sitzman" --model_storage_path "Saliency_Detection/ATSal/attention/weights" --batch_size 80
```

This will result in a trained model that will be stored in the above "model_storage_path". In case that you wish to skip this training step, the weights of this trained attention model (called "ATSal-Attention-Pretrained") are available [here](https://drive.google.com/file/d/13yy42FPF4dS_Gz7wrZjsbbt-kVM3ZpiN/view?usp=sharing)

Following, the attention model was trained using 206 videos from the VR-EyeTracking dataset from [here](https://mtliba.github.io/Reproduced-VR-EyeTracking/); 140 of them were used for training (listed [here](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/VR-EyeTracking/training_data_split.txt)) and the remaining 66 of them for validation (listed [here](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/VR-EyeTracking/validation_data_split.txt)). As a note, videos "102.mp4" and "131.mp4" were excluded due to limited clarity in their ground-truth saliency maps, while the last frames from a few videos (listed [here](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/VR-EyeTracking/Missing_saliency_erp_frames.txt)) were ignored to ensure matching between the number of ground-truth saliency maps and the number of ERP frames.

To further train the attention model using the above dataset (please keep in mind that the extracted ERP frames and the correponding saliency maps for each video should be placed in a same folder), use the [train.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/ATSal/attention/train.py) script and run the following command:
```
python train.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --dataset "VR-EyeTracking" --model_storage_path "Saliency_Detection/ATSal/attention/weights" --batch_size 10 --weight_decay=1e-5
```

If you wish to use the "ATSal-Attention-Pretrained" model, then store it within the above mentioned "model_storage_path" as "pretrained.pt", and run the following command:
```
python train.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --attention_model "weights/pretrained.pt" --model_storage_path "Saliency_Detection/ATSal/attention/weights" --batch_size 10 --weight_decay=1e-5
```

Regarding the SalEMA Expert of ATSal, we further trained an existing model of it (available [here](https://drive.google.com/file/d/1mY2jAL_T06nUs2c_mPVwdX8LtkILhSTI/view?usp=sharing)) that has been trained using the CMP frames of the same 206 videos from VR-EyeTracking (following the same split of data into training and validation set). During our training, frames presenting the north and south regions of the ERP frames ([here](data/VR-EyeTracking/cmp_frames/poles)) were used to train the SalEMA Expert Poles model, while frames presenting the front, back, right and left regions of the ERP frames ([here](data/VR-EyeTracking/cmp_frames/equator)) were used to train the SalEMA Expert Equator model. To further train the SalEMA Expert, place its pretrained model in the directory "Saliency_Detection/ATSal/expert/weights", use the [TrainExpert.py](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/ATSal/expert) script and run the following commands: 

```
python TrainExpert.py --gpu "cuda:0" --path_to_training_cmp_frames "data/VR-EyeTracking/cmp_frames/equator/training/frames" --path_to_validation_cmp_frames "data/VR-EyeTracking/cmp_frames/equator/training/frames" --clip_size 10 --model_storage_path "Saliency_Detection/ATSal/expert/weights"
python TrainExpert.py --gpu "cuda:0" --path_to_training_cmp_frames "data/VR-EyeTracking/cmp_frames/poles/training/frames" --path_to_validation_cmp_frames "data/VR-EyeTracking/cmp_frames/poles/training/frames" --clip_size 80 --model_storage_path "Saliency_Detection/ATSal/expert/weights"
```

Finally, to evaluate the fully-trained ATSal method on the VR-EyeTracking dataset and extract the saliency maps for the testing videos, download the full-trained attention model [here](https://drive.google.com/file/d/1Ke-Ad7lwME6kZdaW8_PUCU7MNklaeJRA/view?usp=sharing), the SalEMA Equator model [here](https://drive.google.com/file/d/1P57U1hZLXAUiwBThq-T-65tZzMG6cm_l/view?usp=sharing), the SalEMA Poles model [here](https://drive.google.com/file/d/1X65FopLF1-m2YtCWC4u0R68HDq0xV3CM/view?usp=sharing), place it in the [weights](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/ATSal/weights) directory, use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/ATSal/inference.py) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --load_gt "True" --dataset "VR-EyeTracking" --path_to_extracted_saliency_maps "data/VR-EyeTracking/extracted_saliency_maps"
```
Moreover, to evaluate the fully-trained ATSal method on the Sports-360 dataset and extract the saliency maps for the testing videos, use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/ATSal/inference.py) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "../Path/360_Saliency_dataset_2018ECCV" --load_gt "True" --dataset "Sports-360" --path_to_extracted_saliency_maps "...data/Sports-360/extracted_saliency_maps"
```

#### SST-Sal method

For the training of the SST-Sal method, we used only a subset of the videos in the VR-EyeTracking dataset, that were captured by a static camera. In total, we used 92 videos for training (listed [here](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/Static-VR-EyeTracking/train_split.txt)) and 55 videos (listed [here](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/Static-VR-EyeTracking/val_split.txt)) for validation. To train the SST-Sal method, use the [train.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/SST-Sal/train.py) script and run the following command: 
```
python train.py --gpu "cuda:0" --hidden_layers 9 --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --model_storage_path "Saliency_Detection/SST-Sal/weights"
```

To evaluate the performance of the trained SST-Sal method on the VR-EyeTracking dataset and extract the saliency maps for the testing videos, download the model [here](https://drive.google.com/file/d/1ANV8Erq2wZpjNRxMgbh4IrWH26XWxCWJ/view?usp=sharing), place it [here](https://github.com/IDT-ITI/CA-SUM-360/tree/main/Saliency_Detection/SST-Sal/weights), use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/SST-Sal/inference.py) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "data/VR-EyeTracking/erp_frames/frames" --load_gt "True" --dataset "VR-EyeTracking"
```
To evaluate the performance of the trained SST-Sal method on the Sports-360 dataset dataset and extract the saliency maps for the testing videos, use the [inference.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Saliency_Detection/SST-Sal/inference.py) script and run the following command:
```
python inference.py --gpu "cuda:0" --path_to_ERP_frames "../Path/360_Saliency_dataset_2018ECCV" --load_gt "True" --dataset "Sports-360"
```

### Salient event detection and 2D video production

Given the extracted saliency maps for the videos of the VR-EyeTracking and Sports-360 datasets, to detect salient events and produced the conventional 2D video that presents these events, use the [main.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/2D_Video_Production/main.py) script and run the following command:
```
python main.py --path_to_ERP_frames ".../data/<dataset-name>/erp_frames" --path_to_extracted_saliency_maps "...data/<dataset-name>/extracted_saliency_maps" --intensity_value 150 --dbscan_distance 1.2 --spatial_distance 100 --fill_loss 100
```
The produced MPEG-4 video files and the computed saliency scores for their frames, will be stored [here](https://github.com/IDT-ITI/CA-SUM-360/tree/main/2D_Video_Production)


### Video summarization

To train the utilized video summarization method we employed 100 conventional 2D videos that were produced after following the previously described processing steps. These videos were created after processing 46 and 19 360-degrees videos of the VR-EyeTracking and Sports-360 datasets, respectively, that were captured using a fixed camera, and 11, 18 and 6 360-degrees videos of the VR-EyeTracking, Sports-360 and Salient360! datasets, respectively, that were captured by a moving camera. The created dataset was divided into a training set (80% of the video samples) and a testing set (the remaining 20% of the video samples), as show in the relevant [json file](https://github.com/IDT-ITI/CA-SUM-360/blob/main/data/Video-Summarization/data_split.json).

For training the method, use the [main.py](https://github.com/IDT-ITI/CA-SUM-360/blob/main/Video_Summarization/model/main.py) script and run the following command (alternatively, open a terminal, load the virtual environment and run "bash run_360videosumm.sh"):
```bash
for sigma in $(seq 0.5 0.1 0.9); do
    python model/main.py --reg_factor '$sigma'
done
```
where `$sigma` refers to the length regularization factor, a hyper-parameter of the utilized method that relates to the length of the generated summary.

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

After each training epoch the algorithm stores the parameters of the trained model and performs an evaluation step, where it uses the model to compute the importance scores for the frames of each video of the test set. These scores are used to select a well-trained model of the summarization method, based on transductive inference. In particular, after the end of the training process, the selection of a well-trained model is based on a two-step process. First, we keep one trained model per considered value for the length regularization factor sigma, by selecting the model (i.e., the epoch) that minimizes the training loss. Then, we choose the best-performing model (i.e., the sigma value) through a mechanism that involves a fully-untrained model of the architecture and is based on transductive inference. To automatically select a well-trained model, define:
 - the [`base_path`](evaluation/evaluate_factor.sh#L7) in [`evaluate_factor`](evaluation/evaluate_factor.sh),
 - the [`base_path`](evaluation/choose_best_model.py#L11) in [`choose_best_model`](evaluation/choose_best_model.py),

and run [`evaluate_exp.sh`](evaluation/evaluate_exp.sh) via
```bash
sh evaluation/evaluate_exp.sh '$exp_num' '$dataset'
```
where, `$exp_num` is the number of the current evaluated experiment, and `$dataset` refers to the dataset being used (should be set as 360VideoSumm).

For further details about the adopted structure of directories in our implementation, please check line [#7](evaluation/evaluate_factor.sh#L7) and line [#13](evaluation/evaluate_factor.sh#L12) of [`evaluate_factor.sh`](evaluation/evaluate_factor.sh). </div>

Finally, the selected model (indicated by the value of the sigma factor and the training epoch) can be used for creating the summaries of the test videos. For this, define the [`model_path`](Video_Summarization/inference/inference.py#71), the [`split_file`](Video_Summarization/inference/inference.py#74) and the [`dataset_path`](Video_Summarization/inference/inference.py#80) in [`inference`](Video_Summarization/inference/inference.py), and run the following command:
```
python inference/inference.py
```
The output of this process indicates the fragments and frames of each testing video, that should be used for building the video summaries.

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


