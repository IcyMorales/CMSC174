import cv2
import numpy as np

# Load the images
image1 = cv2.imread('sampleImages/image1.jpg')
image2 = cv2.imread('sampleImages/image2.jpg')

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# Check if keypoints were detected
if descriptors1 is None or descriptors2 is None:
    print("Error: No keypoints detected in one or both images.")
    exit()

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance (lower distance is better)
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix using RANSAC
homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Print the estimated homography matrix
print("Estimated Homography Matrix:")
print(homography)

# Draw matches
matches_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# for checking
cv2.imwrite('Matched Keypoints.jpg', matches_img)

# Show the matched images
cv2.imshow("Matched Keypoints", matches_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
