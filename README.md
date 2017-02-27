# Parking-Search-Guidance
This project aims to make an intelligent parking system.  
* Firstly,we find the vacant lot by **image subtraction** and then by making contours around that lot we get the coordinates of the vacant parking lot.  
* Then our task is to determine the car's postion and it's rotation angle. This task is done using [ORB](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_orb/py_orb.html) algorithm.
* Our next task is to guide the vehicle intelligently from source to destination point.

###Dependencies:
1. Numpy
2. Matplotlib
3. OpenCV
