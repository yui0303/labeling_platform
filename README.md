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
	- checkpoints: save lastest trained model progress
	- log: logs of training procedure, including each step of loss (use tensorboard to visualize) 
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
		│     └───1.jpg
		│     └───2.jpg
		│     └───3.jpg
		│     │  ...
		│
		└───with_annotation  
		│   └───test
		│   │  └───a1.jpg
		│   │  └───a1.xml
		│   │  └───a2.jpg
		│   │  └───a2.xml
		│   │  │  ...
		│   │
		│   └───train
		│   │  └───b1.jpg
		│   │  └───b1.xml
		│   │  └───b2.jpg
		│   │  └───b2.xml
		│   │  │  ...
		```
		
		If you hasn't decided which are the test dataset, put all labeled images in train directory(with_annotation).  
		And Set the parameter for test dataset later.
		e.g.
		```
		prepare_data
		│
		└───without_annotation
		│   │   
		│   └───train
		│     └───1.jpg
		│     └───2.jpg
		│     └───3.jpg
		│     │  ...
		│
		└───with_annotation  
		│   └───test
		│   │
		│   └───train
		│     └───a1.jpg
		│     └───a1.xml
		│     └───a2.jpg
		│     └───a2.xml
		│     └───b1.jpg
		│     └───b1.xml
		│     └───b2.jpg
		│     └───b2.xml
		│     │  ...
		```
    		
+ Start labeling
	1. Preapre the dataset and put them to the right directory as above  
	2. Open main.html file  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/3.JPG?raw=true)
	3. Change the website download destination(to txtArea)  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/4.JPG?raw=true)
	4. Choose the without annotation directory and press start button  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/5.JPG?raw=true)
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/6.JPG?raw=true)
	5. Now start labeling  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/7.JPG?raw=true)
	6. If there aren't class you want, type class name and insert to the list  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/8.JPG?raw=true)
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/9.JPG?raw=true)
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/10.JPG?raw=true)
	7. If the image is finish, click the save data to file button  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/11.JPG?raw=true)
	8. Though the button can switch the image(Note. If the image that you have labeled, do not label it again)  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/12.JPG?raw=true)
	9. If you have done the labeling, click the train button and go to another page for training  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/13.JPG?raw=true)
	10. Set the parameters and start training (Training process will print in the anaconda terminal)  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/14.JPG?raw=true)
	11. After training, back to label page, reload(F5) the website and do again previous steps  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/15.JPG?raw=true)
	12. After training, you can press predict button to see the result of the training  
		Note. prediction is for the images that haven't been labeled yet.  
	![alt text](https://github.com/yui0303/labeling_platform/blob/main/src/ppt/16.JPG?raw=true)
	13. Keep doing the steps above until you satisfy the result  
	
## Resources
+ https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 (For yolo training)
+ https://github.com/AlexeyAB/darknet (For initial yolo weight)