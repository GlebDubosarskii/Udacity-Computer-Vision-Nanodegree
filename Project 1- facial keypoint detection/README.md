## Project Overview

This project consists of localizing faces in the image and finding 66 facial points. Haar Cascades is used to localize faces (see 3. Facial Keypoint Detection, Complete Pipeline.ipynb). Next, the images of the faces are passed through the neural network that predicts the positions of the facial points. The network training takes place in 2. Define the Network Architecture.ipynb. The training and test data are located in the data folder. Models.py file describes the architecture of the network. I used the Udacity environment to train the network, so you need to modify the second and third notebooks to run my code. You need to change the path to data folder (it shouldn't be too long, this creates an error!) and remove the following lines from the second notebook:
from workspace_utils import active_session
with active_session () :.



![obamas](https://github.com/GlebDubosarskii/Udacity-Computer-Vision-Nanodegree/blob/main/Project%201-%20facial%20keypoint%20detection/images/obamas.jpg)

![detected_faces](https://github.com/GlebDubosarskii/Udacity-Computer-Vision-Nanodegree/blob/main/Project%201-%20facial%20keypoint%20detection/images/detected_faces.png)

![detected_points1](https://github.com/GlebDubosarskii/Udacity-Computer-Vision-Nanodegree/blob/main/Project%201-%20facial%20keypoint%20detection/images/detected_points1.png)

![detected_points2](https://github.com/GlebDubosarskii/Udacity-Computer-Vision-Nanodegree/blob/main/Project%201-%20facial%20keypoint%20detection/images/detected_points2.png)
