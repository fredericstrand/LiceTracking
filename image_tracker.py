import cv2 as cv
import numpy as np

image = cv.imread("images/particles1.png", cv.IMREAD_GRAYSCALE)

image = cv.GaussianBlur(image, (5, 5), 0)
image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

threshold = cv.adaptiveThreshold(
    image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV, 5, 3
)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)

edges = cv.Canny(image, 30, 100)
threshold = cv.bitwise_or(threshold, edges)

contours, _ = cv.findContours(
    threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

small_objects = []
for contour in contours:
    area = cv.contourArea(contour)
    if 2 < area < 500:
        small_objects.append(contour)

output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
cv.drawContours(output_image, small_objects, -1, (0, 0, 255), 1)

cv.imshow("Detected Objects", output_image)
cv.waitKey(0)
cv.destroyAllWindows()
