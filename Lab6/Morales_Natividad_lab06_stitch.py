import cv2
import numpy as np

# List of image filenames
image_files = ['vol1.jpeg', 'vol3.jpeg']

# Load images
images = [cv2.imread(img) for img in image_files]

# Check if all images loaded
if any(img is None for img in images):
    print("Error: One or more images could not be loaded.")
    exit()

# Convert images to grayscale
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Initialize SIFT feature detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for all images
keypoints = []
descriptors = []
for gray in gray_images:
    kp, des = sift.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

# Initialize feature matcher (Brute-Force Matcher)
bf = cv2.BFMatcher()

# Store homographies
homographies = [np.eye(3)]  # Identity matrix for the first image

# Compute homographies between consecutive images
for i in range(len(images) - 1):
    matches = bf.match(descriptors[i], descriptors[i + 1])
    matches = sorted(matches, key=lambda x: x.distance)[:80]  # Take top matches

    # Extract matched keypoints
    src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print(f"Error: Homography between image {i} and {i+1} could not be computed.")
        exit()
    
    # Store homography (cumulative transformation)
    homographies.append(homographies[-1] @ H)

# Warp images to align them
height, width = images[0].shape[:2]
canvas_width = width * len(images)
canvas_height = height

# Create a blank canvas
panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

# Initialize the first image
warped_img = cv2.warpPerspective(images[0], homographies[0], (canvas_width, canvas_height))

# Copy the first image onto the panorama
panorama[:height, :width] = warped_img[:height, :width]

# Blend images using alpha blending
alpha = 0.5  # Blending factor

for i in range(1, len(images)):
    warped_img = cv2.warpPerspective(images[i], homographies[i], (canvas_width, canvas_height))
    
    # Create a mask for non-black areas
    mask = (warped_img > 0)

    # Blend the images
    panorama[mask] = cv2.addWeighted(panorama, alpha, warped_img, 1 - alpha, 0)[mask]

# Save the blended panorama
cv2.imwrite('Panorama.jpg', panorama)

# Show the final blended panorama
cv2.imshow('Blended Panorama', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
