import cv2
import numpy as np 

bev_shape = [256, 512]

pad = 100
src = np.array([[78, 245], [235, 156], [279, 156], [401, 245]], dtype=np.float32)
dst = np.array([[pad, bev_shape[1]], [pad, 0], [bev_shape[0] - pad, 0], [bev_shape[0] -  pad, bev_shape[1]]], dtype=np.float32)


def bird_eye_transform(image):
	transform = cv2.getPerspectiveTransform(src, dst)
	bev = cv2.warpPerspective(image, transform,  (image.shape[0], image.shape[1]))
	return bev

