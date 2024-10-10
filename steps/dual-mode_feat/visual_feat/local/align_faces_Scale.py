# 带缩放的人脸归一化
import cv2
import numpy as np

#返回检测到人脸的最大矩形框
def get_max_rect(self, image, frame_coordinate):
	rect_max_image = np.array(image)
	arr = np.array(frame_coordinate)
	startX = np.min(arr[:, 0])
	startY = np.min(arr[:, 1])
	endX = np.max(arr[:, 0])
	endY = np.max(arr[:, 1])

	rect_max_image = rect_max_image[startY:endY+1, startX:endX+1]
	return rect_max_image
	
# 根据两眼的倾斜角度对图像进行旋转校正
# 根据两眼在图片中的实际距离和自定义的偏移量进行图片的缩放
def align(image, shape):
	desiredLeftEye = (0.35, 0.35)
	desiredFaceWidth = 256
	desiredFaceHeight = 256

	#左右眼的索引
	lStart, lEnd = 75, 83
	rStart, rEnd = 66, 74

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
	pass
	# image = cv2.imread("test_image.jpeg")
	# # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));plt.show()
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# rects = detector(gray, 2)
	# faceAligned = align(gray, rects[0])
	# plt.imshow(cv2.cvtColor(faceAligned, cv2.COLOR_GRAY2RGB))
	# # plt.show()
	# plt.savefig("after_align.jpeg")

