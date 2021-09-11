# Creating a panaroma of Images using OpenCV and Python - Image Stitching

[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](http://numpy.org) [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)  

**What is Image Stitching or Image Panaroma?**  
![Image Stitching Wiki](https://en.wikipedia.org/wiki/Image_stitching)

**Image Dimensions**  
Original Image dimensions used in code is 1920 x 1504. Due to huge file size constraints and to effectively demonstrate the concept of image stitching, the images have been resized and uploaded.  


**Problem Description**  
Individual images of peanut farm bushes are captured to idenitfy the growth of the canopy in terms of width. They are then stitched as a long panaroma image which is used to study and understand efficient use of fertilizers to boost peanut plant growth. The project was done in colloboration with USDA.

**Methods**  
All images are captured using Microsoft Kinect which is mounted on a RC cart. OpenCV and Python is then used to develop the Image stitching algorithm. techniques used include SIFT, Poisson Blending and Homography Matrix generation.  

**Running the code:**
* Download the python and image files and execute *run_main.py*.


**Input Images**  
It would be difficult to display each static image, so a GIF of all the input images are created. The bottom black portion is the RGB Depth captured apart from the top colored RGB image.

![Input](https://github.com/ashwin4ever/Computer-Vision/blob/main/Image%20Stitching/panaroma_input.gif)  


**Output Images**  

![Output](https://github.com/ashwin4ever/Computer-Vision/blob/main/Image%20Stitching/panaroma_output.gif)








