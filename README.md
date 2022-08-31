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

	├─prepare_data   
	│  ├─without_annotation	(For about to label the image)  
	│  │  └─train  
	│  │          
	│  └─with_annotation  
	│      ├─test	(For validation)  
	│      └─train 	(For training)  
	
	.
	+-- _config.yml
	+-- _drafts
	|   +-- begin-with-the-crazy-ideas.textile
	|   +-- on-simplicity-in-technology.markdown
	+-- _includes
	|   +-- footer.html
	|   +-- header.html
	+-- _layouts
	|   +-- default.html
	|   +-- post.html
	+-- _posts
	|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
	|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
	+-- _data
	|   +-- members.yml
	+-- _site
	+-- index.html
	For example,

## Resources
+ https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 (For yolo training)
+ https://github.com/AlexeyAB/darknet (For initial yolo weight)