import cv2
import numpy as np


class ImageAligner:

    def __init__(self, num_of_features=5000):
        self.num_of_features = num_of_features

    def perform_homography(self, prev_img, curr_img):
        # Convert to grayscale.
        new_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        ref_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        height, width = ref_img.shape

        # Create ORB detector with 5000 features.
        orb_detector = cv2.ORB_create(self.num_of_features)

        # Find keypoints and descriptors.
        # The first arg is the image, second arg is the mask
        # (which is not reqiured in this case).
        kp1, d1 = orb_detector.detectAndCompute(new_img, None)
        kp2, d2 = orb_detector.detectAndCompute(ref_img, None)

        # Match features between the two images.
        # We create a Brute Force matcher with
        # Hamming distance as measurement mode.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the two sets of descriptors.
        matches = matcher.match(d1, d2)

        # Sort matches on the basis of their Hamming distance.
        matches.sort(key=lambda x: x.distance)

        # Take the top 90 % matches forward.
        matches = matches[:int(len(matches)*90)]
        no_of_matches = len(matches)

        # Define empty matrices of shape no_of_matches * 2.
        p1 = np.zeros((no_of_matches, 2))
        p2 = np.zeros((no_of_matches, 2))

        for i in range(len(matches)):
            p1[i, :] = kp1[matches[i].queryIdx].pt
            p2[i, :] = kp2[matches[i].trainIdx].pt

        # Find the homography matrix.
        homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

        # Use this matrix to transform the
        # colored image wrt the reference image.
        transformed_img = cv2.warpPerspective(curr_img, homography, (width, height))

        #cv2.imshow("transformed", transformed_img)
        #cv2.waitKey(1)

        return transformed_img

    def resize(self, img, dim=(300, 800)):
        """
        img = prev_bbox
        """
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        #cv2.imshow("bbox", img)
        #cv2.waitKey(1)

        assert resized.shape == (dim[1], dim[0], 3), "Image not right size"

        return resized

"""
    def resize_image(self, img, size=(500, 800)):

        h, w, c = img.shape

        #if h == w:
        #    return cv2.resize(img, size, cv2.INTER_AREA)

        dif = h if h > w else w

        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        resized = cv2.resize(mask, size, interpolation)

        cv2.imshow("resized", resized)
        cv2.waitKey(1)

        return resized

 def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h

    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r , im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv2.resize(img, (im_w_r , im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))

    return img


cv2.imshow("resized", crop_and_resize(img, 800, 1200))
cv2.waitKey(1)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



cv2.imshow("resized", image_resize(img, width=300, height=800))
cv2.waitKey(1)

import imutils
imutils.resize(img, width=300, height=800).shape

small = cv2.resize(img, (50, 200), interpolation = cv2.INTER_AREA)

cv2.imshow("resized", cv2.resize(small, (300, 800), interpolation = cv2.INTER_AREA))
cv2.imshow("resized", cv2.resize(img, (50, 200), interpolation = cv2.INTER_AREA))
cv2.waitKey(1)
"""
