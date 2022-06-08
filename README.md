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







