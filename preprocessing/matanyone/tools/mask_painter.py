import cv2
import torch
import numpy as np
from PIL import Image
import copy
import time


def colormap(rgb=True):
	color_list = np.array(
		[
			0.000, 0.000, 0.000,
			1.000, 1.000, 1.000,
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list


color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def vis_add_mask(image, background_mask, contour_mask, background_color, contour_color, background_alpha, contour_alpha):
	background_color = np.array(background_color)
	contour_color = np.array(contour_color)

	# background_mask = 1 - background_mask
	# contour_mask = 1 - contour_mask

	for i in range(3):
		image[:, :, i] = image[:, :, i] * (1-background_alpha+background_mask*background_alpha) \
			+ background_color[i] * (background_alpha-background_mask*background_alpha)
		
		image[:, :, i] = image[:, :, i] * (1-contour_alpha+contour_mask*contour_alpha) \
			+ contour_color[i] * (contour_alpha-contour_mask*contour_alpha)

	return image.astype('uint8')


def mask_generator_00(mask, background_radius, contour_radius):
	# no background width when '00'
	# distance map
	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.

	return mask, contour_mask


def mask_generator_01(mask, background_radius, contour_radius):
	# no background width when '00'
	# distance map
	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	return mask, contour_mask


def mask_generator_10(mask, background_radius, contour_radius):
	# distance map
	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# .....:::::!!!!!
	background_mask = np.clip(dist_map, -background_radius, background_radius)
	background_mask = (background_mask - np.min(background_mask))
	background_mask = background_mask / np.max(background_mask)
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	contour_mask[contour_mask>0.5] = 1.
	return background_mask, contour_mask


def mask_generator_11(mask, background_radius, contour_radius):
	# distance map
	dist_transform_fore = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
	dist_transform_back = cv2.distanceTransform(1-mask, cv2.DIST_L2, 3)
	dist_map = dist_transform_fore - dist_transform_back
	# .....:::::!!!!!
	background_mask = np.clip(dist_map, -background_radius, background_radius)
	background_mask = (background_mask - np.min(background_mask))
	background_mask = background_mask / np.max(background_mask)
	# ...:::!!!:::...
	contour_radius += 2
	contour_mask = np.abs(np.clip(dist_map, -contour_radius, contour_radius))
	contour_mask = contour_mask / np.max(contour_mask)
	return background_mask, contour_mask


def mask_painter(input_image, input_mask, background_alpha=0.5, background_blur_radius=7, contour_width=3, contour_color=3, contour_alpha=1, mode='11'):
	"""
	Input:
	input_image: numpy array
	input_mask: numpy array
	background_alpha: transparency of background, [0, 1], 1: all black, 0: do nothing
	background_blur_radius: radius of background blur, must be odd number
	contour_width: width of mask contour, must be odd number
	contour_color: color index (in color map) of mask contour, 0: black, 1: white, >1: others
	contour_alpha: transparency of mask contour, [0, 1], if 0: no contour highlighted
	mode: painting mode, '00', no blur, '01' only blur contour, '10' only blur background, '11' blur both

	Output:
	painted_image: numpy array
	"""
	assert input_image.shape[:2] == input_mask.shape, 'different shape'
	assert background_blur_radius % 2 * contour_width % 2 > 0, 'background_blur_radius and contour_width must be ODD'
	assert mode in ['00', '01', '10', '11'], 'mode should be 00, 01, 10, or 11'

	# downsample input image and mask
	width, height = input_image.shape[0], input_image.shape[1]
	res = 1024
	ratio = min(1.0 * res / max(width, height), 1.0)  
	input_image = cv2.resize(input_image, (int(height*ratio), int(width*ratio)))
	input_mask = cv2.resize(input_mask, (int(height*ratio), int(width*ratio)))
	
	# 0: background, 1: foreground
	msk = np.clip(input_mask, 0, 1)

	# generate masks for background and contour pixels
	background_radius = (background_blur_radius - 1) // 2
	contour_radius = (contour_width - 1) // 2
	generator_dict = {'00':mask_generator_00, '01':mask_generator_01, '10':mask_generator_10, '11':mask_generator_11}
	background_mask, contour_mask = generator_dict[mode](msk, background_radius, contour_radius)

	# paint
	painted_image = vis_add_mask\
		(input_image, background_mask, contour_mask, color_list[0], color_list[contour_color], background_alpha, contour_alpha)	# black for background

	return painted_image


if __name__ == '__main__':
	
	background_alpha = 0.7  	# transparency of background 1: all black, 0: do nothing
	background_blur_radius = 31	# radius of background blur, must be odd number
	contour_width = 11       	# contour width, must be odd number
	contour_color = 3      		# id in color map, 0: black, 1: white, >1: others
	contour_alpha = 1       	# transparency of background, 0: no contour highlighted

	# load input image and mask
	input_image = np.array(Image.open('./test_img/painter_input_image.jpg').convert('RGB'))
	input_mask = np.array(Image.open('./test_img/painter_input_mask.jpg').convert('P'))
	
	# paint
	overall_time_1 = 0
	overall_time_2 = 0
	overall_time_3 = 0
	overall_time_4 = 0
	overall_time_5 = 0
	
	for i in range(50):
		t2 = time.time()
		painted_image_00 = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha, mode='00')
		e2 = time.time()

		t3 = time.time()
		painted_image_10 = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha, mode='10')
		e3 = time.time()

		t1 = time.time()
		painted_image = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha)
		e1 = time.time()

		t4 = time.time()
		painted_image_01 = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha, mode='01')
		e4 = time.time()

		t5 = time.time()
		painted_image_11 = mask_painter(input_image, input_mask, background_alpha, background_blur_radius, contour_width, contour_color, contour_alpha, mode='11')
		e5 = time.time()

		overall_time_1 += (e1 - t1)
		overall_time_2 += (e2 - t2)
		overall_time_3 += (e3 - t3)
		overall_time_4 += (e4 - t4)
		overall_time_5 += (e5 - t5)

	print(f'average time w gaussian: {overall_time_1/50}')
	print(f'average time w/o gaussian00: {overall_time_2/50}')
	print(f'average time w/o gaussian10: {overall_time_3/50}')
	print(f'average time w/o gaussian01: {overall_time_4/50}')
	print(f'average time w/o gaussian11: {overall_time_5/50}')

	# save
	painted_image_00 = Image.fromarray(painted_image_00)
	painted_image_00.save('./test_img/painter_output_image_00.png')

	painted_image_10 = Image.fromarray(painted_image_10)
	painted_image_10.save('./test_img/painter_output_image_10.png')

	painted_image_01 = Image.fromarray(painted_image_01)
	painted_image_01.save('./test_img/painter_output_image_01.png')

	painted_image_11 = Image.fromarray(painted_image_11)
	painted_image_11.save('./test_img/painter_output_image_11.png')
