# LiceTracking

## Project description
This project aims to track Salom Lice's response to light using OpenCV and Python. By detecting and tracking their movement and their speed we aim to gain insight in how light may affect them. 

The program will both output a video with contours around the particles and a plot for some random particles. 

## Functionality

**Folder**
If not already created, the program will create a folder for its plots.

**Filters**
> cvtColor: We use this filter to turn the video into grayscale (BGR2GRAY)
> GaussianBlur: Smoothen the image to reduce noise
> bilateralFilter: Smoothen the image, while conserving edges. (removed)
> normalize: Scale the data to a specific range. Might be needed for machine learning and coming apllications. (removed)
> Threshold: Adaptive Gaussian Thresholding to convert image to binary, and MORPH_ELLIPSE to enhance or supress features.
> Contours: Gets only the outer most points, and collapses lines to save computation.

## Installation
**Requirements**
- Python3
- opencv (cv2)
- matplotlib
- os

**Clone the repository**
> link: 

