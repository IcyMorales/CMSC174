import cv2
import numpy as np

def stitchImage(img1, img2):
    # Load the images
    image1 = img1
    image2 = img2

    # Check if images are loaded correctly
    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded.")
        exit()

    # Convert images to grayscale
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
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 80
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Check if homography matrix was found
    if homography is None:
        print("Error: Homography could not be computed.")
        exit()

    # Warp the first image using the homography
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

    # Blending the warped image with the second image using alpha blending
    alpha = 0.5  # Blending factor
    blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

    return blended_image
'''
    # for checking 
    cv2.imwrite('Blended Image.jpg', blended_image)

    # Display the blended image
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
