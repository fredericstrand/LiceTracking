import cv2 as cv
import numpy as np

image = cv.imread("images/kaffe_image_1.jpg", cv.IMREAD_GRAYSCALE)

image = cv.GaussianBlur(image, (11, 11), 0)
image = cv.bilateralFilter(image, 11, 75, 75 )
image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

threshold = cv.adaptiveThreshold(
    image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV, 21, 2
)

edges = cv.Canny(image, 30, 100)
threshold = cv.bitwise_or(threshold, edges)

contours, _ = cv.findContours(
    threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

objects = []
for contour in contours:
    area = cv.contourArea(contour)
    if  50 < area < 750:
        objects.append(contour)

output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
cv.drawContours(output_image, objects, -1, (0, 0, 255), 1)

print(len(objects))

cv.imshow("Detected Objects", output_image)
cv.waitKey(0)
cv.destroyAllWindows()
