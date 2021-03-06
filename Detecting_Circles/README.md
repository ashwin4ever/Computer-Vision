

## This code implements semi-automatic labelling of Gasometer image markings. Reference to the problem and original approach can be viewed at:
[Feature extraction and machine learning techniques for identifying historic urban environmental hazards: New methods to locate lost fossil fuel infrastructure in US cities](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0255507)

[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](http://numpy.org) [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)  

**Probelm:**  
To detect circular gasometer markings in Sanborn Fire Insurance Maps.

**Data Information:**
Sanborn Fire Insurance Maps provided by Library of Congress  
- **ROI**: This folder contains the regions of interest extracted after applying Hough Transforms and is used for clustering

**Code information:**  
- [k_means_sanborn.ipynb](https://github.com/ashwin4ever/Computer-Vision/blob/main/Detecting_Circles/k_means_sanborn.ipynb): Identifies and clusters images as either having Gasometer marking or not. This is a binary classification problem with 2 classes. **Cluster 0:** Positive class and **Cluster 1:** Negative class 
- [detect_sanborn_gasometers.ipynb](https://github.com/ashwin4ever/Computer-Vision/blob/main/Detecting_Circles/detect_sanborn_gasometers.ipynb): This detects circular portions from the image maps using Hough Transforms 
- [sanborn_maps_download.ipynb](https://github.com/ashwin4ever/Computer-Vision/blob/main/Detecting_Circles/sanborn_maps_download.ipynb): Code to download Sanborn Image maps

