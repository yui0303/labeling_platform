# Labeling Platform

## Preface
It is a website based labeling platform with yolo model training.  
There might sime bug in the system.  
If there are some bugs or problems about the platform, welcome to raise the issue.  

The platform only support  
	1. Google Chorme (check whether it is the lastest version)  
	2. XML format label  

## Set the environment
+ Install Anaconda
	- conda create --name myenv python=3.7.13
	- pip install -r requirements.txt
	- Ignore the error prompt as shown below if you have counter
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/version_error.jpg?raw=true)
+ downloads initial yolo weight:
	- https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
	- put the weight file under model_data directory

## Operating manual
+ Iroduction of Directories

	```
	prepare_data
	│   README.md
	│   file001.txt    
	│
	└───without_annotation	(For about to label the image)  
	│   │   
	│   └───train
	│
	└───with_annotation
		│   
		└───train (For validation)
		│
		└───train (For training)
	```


## Resources
+ https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 (For yolo training)
+ https://github.com/AlexeyAB/darknet (For initial yolo weight)