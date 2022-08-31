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
	- Prepare Data
		```
		prepare_data
		│
		└───without_annotation	(For images about to label) 
		│   └───train
		└───with_annotation
			└───test (For validation)
			└───train (For training)
			
		```
		For example, 
		```
		prepare_data
		│
		└───without_annotation
		│   │   
		│   └───train
		│   	└───1.jpg
		│   	└───2.jpg
		│   	└───3.jpg
		│		│	...
		│
		└───with_annotation  
		│	└───test
		│	│	└───a1.jpg
		│	│	└───a1.xml
		│	│	└───a2.jpg
		│	│	└───a2.xml
		│	│	│	...
		│   │
		│	└───train
		│	│	└───b1.jpg
		│	│	└───b1.xml
		│	│	└───b2.jpg
		│	│	└───b2.xml
		│	│	│	...
		```
		
		If you not sure which should be the test dataset, put all labeled images in train directory(with_annotation).  
		And Set the parameter for test dataset later.
		e.g.
		```
		prepare_data
		│
		└───without_annotation
		│   │   
		│   └───train
		│		└───1.jpg
		│		└───2.jpg
		│		└───3.jpg
		│		│	...
		│
		└───with_annotation  
		│	└───test
		│	│
		│	└───train
		│	  └───a1.jpg
		│	  └───a1.xml
		│	  └───a2.jpg
		│	  └───a2.xml
		│	  └───b1.jpg
		│	  └───b1.xml
		│	  └───b2.jpg
		│	  └───b2.xml
		│	  │	...
		```
	
## Resources
+ https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 (For yolo training)
+ https://github.com/AlexeyAB/darknet (For initial yolo weight)