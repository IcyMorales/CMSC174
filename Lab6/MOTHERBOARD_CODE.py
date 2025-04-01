import cv2
import numpy as np

#DONOTE THAT IMAGES USED ARE RENAMED IN THE /DATA as IMG1,IMG2...

def stitch_images(img1, img2):
    """Stitches two images together using feature matching and homography."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Feature detection using SIFT
    sift = cv2.SIFT_create(nfeatures=1000)
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Brute-Force Matcher with KNN
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's Ratio Test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < 4:
        print("Not enough matches to compute homography.")
        return None

    # Get matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    # Get the new canvas size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners1, H)
    
    all_corners = np.concatenate((warped_corners, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)), axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Translation to fit everything
    translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H = translation_matrix @ H

    # Warp image1
    final_width = x_max - x_min
    final_height = y_max - y_min
    stitched = cv2.warpPerspective(img1, H, (final_width, final_height))

    # Paste image2 onto the stitched canvas
    stitched[-y_min:h2 - y_min, -x_min:w2 - x_min] = img2

    return stitched


# Load images
image1 = cv2.imread('data/img1.jpg')
image2 = cv2.imread('data/img2.jpg')
image3 = cv2.imread('data/img3.jpg')
image4 = cv2.imread('data/img4.jpg')

# Step 1: Stitch (image1, image3) → result13
result12 = stitch_images(image1, image3)
cv2.imwrite("result12.jpg", result12)

# Step 2: Stitch (image2, image4) → result24
result34 = stitch_images(image2, image4)
cv2.imwrite("result34.jpg", result34)

# Step 3: Merge result12 and result34
final_result = stitch_images(result12, result34)

# Save and display final image
cv2.imshow("Final Stitched Image", final_result)
cv2.imwrite("Final_Stitched_Image.jpg", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
