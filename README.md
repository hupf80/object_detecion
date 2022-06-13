# object_detecion


Package for the Floribot Object detection.


The following other Packages are requred: https://github.com/hupf80/boundingbox_msgs

Please Note:

For installation Structures follow the instructions of:

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html to install TenorRT. For the testcases here, TensorRT 8 is used.

Also it is neccessary to install CUDA (https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

and cuDNN (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)


please use only with the Floribot Simulation.


Currently it is not ina Docker enviorment.

If you don't have a CUDA Graphic Card theis would never work.

Ask the Contributer for the weights file. Which has to be inserted into the configs/yolov5-6.0  

https://drive.google.com/file/d/1vysAi7l8uDr88RVkjP_680GRppttRq8t/view?usp=sharing for Downloading the weights file.

Please copy the weights file into configs/yolov5-6.0 as described above.

##Instructions for Tracker

The Object Detection has a simple nearest-neighbour object-tracking algorithm implemented.

In the launch file there are settings, which can be adjusted.

1. ´conv_tsh´ which is the convidence treshold for detecting objects
2. ´tracking_closest_tsh_pixel´ which is the treshold, when a object of the same class is still matched in the distance in pixel
3. ´tracking_max_obj_age´ says how old an object can get, when it is not matched with old tracklets.






