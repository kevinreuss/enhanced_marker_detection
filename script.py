import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image

# CONSTANTS #
COLOR_TRESHOLD = 300
PERCENTAGE_TRESHOLD = 0.4
PERCENTAGE_TRESHOLD_CALCULATED = 0.2

def min(my_list, max_value):
	value = max_value
	for i in my_list:
		if i[0] <= value:
			value = i[0]
	return value

def max(my_list, min_value):
	value = min_value
	for i in my_list:
		if i[0] >= value:
			value = i[0]
	return value

def mean(my_list):
	value_sum = 0
	for i in my_list:
		value_sum += i[1]
	return float(value_sum) / float(len(my_list))

def variance(my_list, mean):
	variance = 0
	for i in my_list:
		variance += pow(float(i[1]) - mean, 2)
	variance /= float(len(my_list))
	return variance

def filter(my_list):
	standard_deviation = math.sqrt(variance(my_list, mean(my_list)))
	counter = 0
	for i in my_list:
		if abs(i[1] - standard_deviation) > standard_deviation:
			my_list.pop(counter)
		counter += 1
	return my_list

def main():
	
	input_image = Image.open("input.bmp")
	input_image.load()
	height, width = input_image.size

	# DETERMINE MIN & MAX VALUES #
	min_value = 3 * 255
	max_value = 0
	for column in range(0, width):
		for row in range(0, height):
			value = sum(input_image.getpixel((column, row))) #input_image.getpixel(1, 1)
			if value <= min_value:
				min_value = value
			if value >= max_value:
				max_value = value

	COLOR_TRESHOLD = min_value + ((max_value - min_value) / 2)
	#print COLOR_TRESHOLD

	# CREATE FILTERED IMAGE #
	for column in range(0, width):
		for row in range(0, height):
			value = sum(input_image.getpixel((column, row)))
			if value <= COLOR_TRESHOLD:
				input_image.putpixel((column, row), (0, 0, 0))
			else:
				input_image.putpixel((column, row), (255, 255, 255))

	input_image.save("new.bmp")

	input_image = Image.open("new.bmp")
	input_image.load()
	height, width = input_image.size

	# DETECT RECTANGLE #
	from_left = []
	from_right = []
	from_top = []
	from_bottom = []

	# FROM LEFT #
	for row in range(0, height):
		for column in range(0, width):
			if sum(input_image.getpixel((column, row))) == 0:
				from_left.append([row, column])
				break
	from_left = filter(from_left)
	x_coord = int(mean(from_left))
	y_start = min(from_left, height)
	y_end = max(from_left, 0)
	
	# FROM RIGHT #
	for row in range(0, height):
		for column in range(0, width):
			if sum(input_image.getpixel((width - column - 1, row))) == 0:
				from_right.append([row, width - column])
				break
	from_right = filter(from_right)
	x_coord_two = int(mean(from_right))
	y_start_two = min(from_right, height)
	y_end_two = max(from_right, 0)
	
	y_start = (y_start + y_start_two) / 2
	y_end = (y_end + y_end_two) / 2

	# PRINT RECTANGLE #
	for row in range(y_start - 1, y_end + 1):
		input_image.putpixel((x_coord, row), (255, 0, 0))
	for row in range(y_start - 1, y_end + 1):
		input_image.putpixel((x_coord_two, row), (255, 0, 0))

	for column in range(x_coord, x_coord_two):
		input_image.putpixel((column, y_start - 1), (255, 0, 0))
	for column in range(x_coord, x_coord_two):
		input_image.putpixel((column, y_end), (255, 0, 0))

	# CALCULATE GRID #

	grid_width = 2 + (x_coord_two - x_coord) / 3
	grid_height = (y_end - y_start) / 3

	# PRINT GRID #
	for row in range(y_start - 1, y_end + 1):
		input_image.putpixel((x_coord + grid_height, row), (255, 0, 0))
		input_image.putpixel((x_coord + 2 * grid_height, row), (255, 0, 0))
	for column in range(x_coord, x_coord_two):
		input_image.putpixel((column, y_start + grid_width) , (255, 0, 0))
		input_image.putpixel((column, y_start + 2 * grid_width) , (255, 0, 0))

	input_image.save("new_2.bmp")


	# DETECT PATTERN #
	x_fields = [x_coord, x_coord + grid_width, x_coord + 2 * grid_width, x_coord_two]
	y_fields = [y_start, y_start + grid_height, y_start + 2 * grid_height, y_end]
	fields_percentage = []

	for y in range(len(y_fields) - 1):
		tmp_percentage = []
		for x in range(len(x_fields) - 1):
			x_start = x_fields[x]
			y_start = y_fields[y]
			x_end = x_fields[x + 1]
			y_end = y_fields[y + 1]

			value = 0
			for x_c in range(x_start, x_end):
				for y_c in range(y_start, y_end):
					if sum(input_image.getpixel((x_c, y_c))) == 0:
						value += 1
			amount = (x_end - x_start) * (y_end - y_start)
			percentage = float(value) / float(amount)
			tmp_percentage.append(percentage)
		fields_percentage.append(tmp_percentage[:])

	#print fields_percentage
	#print "-------------------------"

	fields = []
	
	for i in fields_percentage:
		tmp = []
		for j in i:
			if j >= PERCENTAGE_TRESHOLD:
				tmp.append("#")
			else:
				tmp.append(" ")
		fields.append(tmp[:])

	print "_________________"
	print " DETECTED MARKER"
	print "_________________"
	

	# PRINT DETECTED MARKER #
	for f in fields:
		print "|" + f[0] + "|" + f[1] + "|" + f[2] + "|"

	input_image.save("new_3.bmp")

if __name__ == "__main__":
	main()


















































"""
input_image = cv2.imread("input_3.png", 0)

# converting to gray scale
#gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# remove noise
#blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

# convolute with proper kernels
laplacian = cv2.Laplacian(input_image, cv2.CV_64F)
#sobelx = cv2.Sobel(input_image, cv2.CV_64F, 1, 0, ksize = 5)
#sobely = cv2.Sobel(input_image, cv2.CV_64F, 0, 1, ksize = 5)

plt.subplot(2 ,2, 1),plt.imshow(input_image, cmap = "gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])


plt.subplot(2, 2, 2),plt.imshow(laplacian)
plt.title("Laplacian"), plt.xticks([]), plt.yticks([])

#plt.subplot(2, 2, 2),plt.imshow(sobelx, cmap = "gray")
#plt.title("Sobel X"), plt.xticks([]), plt.yticks([])
#plt.subplot(2, 2, 3),plt.imshow(sobely, cmap = "gray")
#plt.title("Sobel Y"), plt.xticks([]), plt.yticks([])

plt.show()
"""
