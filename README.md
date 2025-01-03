# LiceTracking 🐟

## Project description 
This project aims to track Salom Lice's response to light using OpenCV and Python. By detecting and tracking their movement and their speed we aim to gain insight in how light may affect them. 

## Functionality

### Folder structure
If not already created, the program will create a folder for its plots.

### Filters 🎯
- > **cvtColor:** We use this filter to turn the video into grayscale (BGR2GRAY)
- > **GaussianBlur:** Smoothen the image to reduce noise
- > **bilateralFilter:** Smoothen the image, while conserving edges. (removed)
- > **normalize:** Scale the data to a specific range. Might be needed for machine learning and coming apllications. (removed)
- > **Threshold:** Adaptive Gaussian Thresholding to convert image to binary, and MORPH_ELLIPSE to enhance or supress features.
- > **Contours:** Gets only the outer most points, and collapses lines to save computation.
  
### Visualization 📺
The program will both output a video with contours around the particles and a plot for some random particles. 

**Plot of the particles movements**
To number of particles plotted using matplotlib is given by variable num_plotted_particles

**Contour of particles example**
![image](https://github.com/user-attachments/assets/4bb71630-1121-4d88-abbb-daeea080861f)

## Installation
**Requirements**
- Python3
- opencv (cv2)
- matplotlib
- os

**Clone the repository**
> link: https://github.com/fredericstrand/LiceTracking.git

