# Labeling Platform

## Set the environment
+ Install Anaconda
	- conda create --name myenv python=3.7.13
	- pip install -r requirements.txt
	- Ignore the error prompt as shown below if you have counter
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/version_error.jpg?raw=true)
+ downloads initial yolo weight:
	- https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
	- put the weight file under model_data directory

## Start Labeling

## Resources
+ https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 (For yolo training)
+ https://github.com/AlexeyAB/darknet (For initial yolo weight)