# 带缩放的人脸归一化
import sys
from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/mnt/shareEx/lushangjun/Lipspeech/data_process/dlib/shape_predictor_68_face_landmarks.dat')

def face_align(image):
	# 须在外部先转换为灰度图像
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=800)

	# detect faces in the grayscale
	rects = detector(image, 2)
	if len(rects) == 0:
		rects = detector(image, 1)
		if len(rects) == 0:
			rects = detector(image)

	if len(rects) == 0:
		return []
	# The first face detections
	rect = rects[0]
	# align the face
	faceAligned = align(image, rect)

	return faceAligned

def face_align_withmax(image, rect_max):
	# 须在外部先转换为灰度图像
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = imutils.resize(image, width=800)

	# detect faces in the grayscale
	rects = detector(image, 2)
	if len(rects) == 0:
		rects = detector(image, 1)
		if len(rects) == 0:
			rects = detector(image)

	if len(rects) == 0:
		rects = rect_max
	# The first face detections
	rect = rects[0]
	# align the face
	faceAligned = align(image, rect)

	return faceAligned

def align(image, rect):
	desiredLeftEye = (0.35, 0.35)
	desiredFaceWidth = 256
	desiredFaceHeight = 256
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(image, rect)
	shape = imutils.face_utils.shape_to_np(shape)

	# simple hack ;)
	if (len(shape) == 68):
		# extract the left and right eye (x, y)-coordinates
		(lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
		(rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	else:
		(lStart, lEnd) = imutils.face_utils.FACIAL_LANDMARKS_5_IDXS["left_eye"]
		(rStart, rEnd) = imutils.face_utils.FACIAL_LANDMARKS_5_IDXS["right_eye"]

	leftEyePts = shape[lStart:lEnd]
	rightEyePts = shape[rStart:rEnd]

	# compute the center of mass for each eye
	leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
	rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

	# compute the angle between the eye centroids
	dY = rightEyeCenter[1] - leftEyeCenter[1]
	dX = rightEyeCenter[0] - leftEyeCenter[0]
	angle = np.degrees(np.arctan2(dY, dX)) - 180

	# compute the desired right eye x-coordinate based on the
	# desired x-coordinate of the left eye
	desiredRightEyeX = 1.0 - desiredLeftEye[0]

	# determine the scale of the new resulting image by taking
	# the ratio of the distance between eyes in the *current*
	# image to the ratio of distance between eyes in the
	# *desired* image
	dist = np.sqrt((dX ** 2) + (dY ** 2))
	desiredDist = (desiredRightEyeX - desiredLeftEye[0])
	desiredDist *= desiredFaceWidth
	scale = desiredDist / dist

	# compute center (x, y)-coordinates (i.e., the median point)
	# between the two eyes in the input image

	# eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
	# 	(leftEyeCenter[1] + rightEyeCenter[1]) // 2)
	eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) / 2),
				  int((leftEyeCenter[1] + rightEyeCenter[1]) / 2))

	# grab the rotation matrix for rotating and scaling the face
	M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

	# update the translation component of the matrix
	tX = desiredFaceWidth * 0.5
	tY = desiredFaceHeight * desiredLeftEye[1]
	M[0, 2] += (tX - eyesCenter[0])
	M[1, 2] += (tY - eyesCenter[1])

	# apply the affine transformation
	(w, h) = (desiredFaceWidth, desiredFaceHeight)
	output = cv2.warpAffine(image, M, (w, h),
							flags=cv2.INTER_CUBIC)

	# return the aligned face
	return output

if __name__ == '__main__':
	image = cv2.imread("test_image.jpeg")
	# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.show()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 2)
	faceAligned = align(gray, rects[0])
	plt.imshow(cv2.cvtColor(faceAligned, cv2.COLOR_GRAY2RGB))
	# plt.show()
	plt.savefig("after_align.jpeg")

