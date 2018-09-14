import scipy as sp
import numpy as np
from scipy.misc import imread, imresize, imsave, imshow
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# read an image from current directory
img = imread('cat.jpg',)

# print the image shape and its type
print(img.shape, img.dtype)

# tint the image by multiplying it in a weighted array

tint_img = img*[1, 0.95, 0.9]

# show tinted image
imshow(tint_img)
plt.show()

# resize the image to any arbitrary size

resize_img = imresize(tint_img, (300, 270))
# # show resized image
# imshow(resize_img)
# plt.show()

# write the resize and tinted image to the disk
imsave('tint_resize_cat.jpg', resize_img)


# reading matlab files

# creating an array
x = np.array(np.random.randint(0, 5, size=(3, 2)))
print(x)

# compute Euclidean distance between all rows
d = squareform(pdist(x, 'euclidean'))
print(d)

