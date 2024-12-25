import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Ensure the plots directory exists
plots_directory = "plots"
if not os.path.exists(plots_directory):
    os.makedirs(plots_directory)

video_path = "videos/particles_video_1.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

particles = []
particles_movement = {}
particle_id = 0

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (11, 11), 0)
    image = cv.bilateralFilter(image, 11, 75, 75)
    image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)
    threshold = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 5)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    threshold = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)
    edges = cv.Canny(image, 30, 100)
    threshold = cv.bitwise_or(threshold, edges)

    # Find contours and track particles
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    current_particles = []
    for contour in contours:
        area = cv.contourArea(contour)
        if 5 < area < 500:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                current_particles.append((cx, cy, area))

    updated_particles = []
    for cx, cy, area in current_particles:
        min_dist = float('inf')
        matched_id = None
        for particle in particles:
            pid, (px, py), _ = particle
            dist = distance((cx, cy), (px, py))
            if dist < min_dist and dist < 50:
                min_dist = dist
                matched_id = pid

        if matched_id is not None:
            updated_particles.append((matched_id, (cx, cy), area))
            particles_movement[matched_id].append((cx, cy))
        else:
            updated_particles.append((particle_id, (cx, cy), area))
            particles_movement[particle_id] = [(cx, cy)]
            particle_id += 1

    particles = updated_particles
    output_frame = frame.copy()
    cv.drawContours(output_frame, contours, -1, (0, 0, 255), 1)
    cv.imshow("Detected Particles", output_frame)
    if cv.waitKey(1) & 0xFF == ord('c'):
        break

cap.release()
cv.destroyAllWindows()

# Plotting the particles's movement
num_plotted_particles = 10
selected_particles = random.sample(list(particles_movement.keys()), num_plotted_particles)

fig, ax = plt.subplots()
for pid in selected_particles:
    positions = particles_movement[pid]
    xs = [pos[0] for pos in positions]
    ys = [pos[1] for pos in positions]
    ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, alpha=0.7, label=f'Particle {pid}')

ax.set_xlabel('cx')
ax.set_ylabel('cy')
ax.set_title(f'Random Particle Movements Over Time')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(plots_directory, "random_particle_movement.png"))
plt.show()
