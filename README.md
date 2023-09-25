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
