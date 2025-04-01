

import cv2

# Load the images
image1 = cv2.imread('sampleImages/image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('sampleImages/image2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded correctly
if image1 is None or image2 is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Check if keypoints were detected
if descriptors1 is None or descriptors2 is None:
    print("Error: No keypoints detected in one or both images.")
    exit()

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches_bf = bf.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Draw the top N matches
num_matches = min(50, len(matches_bf))  # Ensure we don't exceed available matches
image_matches_bf = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_bf[:num_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Initialize the feature matcher using FLANN matching
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match the descriptors using FLANN matching
matches_flann = flann.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_flann = sorted(matches_flann, key=lambda x: x.distance)

# Draw the top N matches
num_matches_flann = min(50, len(matches_flann))  # Ensure we don't exceed available matches
image_matches_flann = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches_flann], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# for checking
cv2.imwrite('Brute-Force Matching.jpg', image_matches_bf)
cv2.imwrite('FLANN Matchings.jpg', image_matches_flann)

# Display the images with matches
cv2.imshow('Brute-Force Matching', image_matches_bf)
cv2.imshow('FLANN Matching', image_matches_flann)

cv2.waitKey(0)
cv2.destroyAllWindows()
