from DataManager.manager import get_visualization_batch, get_batch
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as im
import scipy.io as sio
import numpy as np

#######################
### Scaling Imports ###
#######################
# from mpl_toolkits.mplot3d.axes3d import Axes3D





STRIDE = 1 # Higher stride means more subsampling (faster rendering, worse quality)


########################
### Helper Functions ###
########################

"""Displays a single image retrieved using get_batch()."""
def demo_cropped():
	images, voxel_lst = get_batch(1)
	image = Image.fromarray(images[0], 'RGB')
	voxels = voxel_lst[0]

	visualize_voxels_cropped(image, voxels, save=True)

def demo(save=False):
	images, voxel_lst, transforms = get_visualization_batch(1)
	image = Image.fromarray(images[0], 'RGB')
	voxels = voxel_lst[0]
	transform = transforms[0]

	visualize_voxels_original(image, voxels, transform, save)


'''Visualizes the voxels on top of the cropped image.

Input:  The cropped image and the voxels.

Output: If save flag set, saves different views of the 
visualization in root. Otherwise, simply opens a window with
the visualization from a side-front view.
'''
def visualize_voxels_cropped(cropped_image, voxels, save=False):
	fig = plt.figure()
	fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

	ax = fig.add_subplot(111, projection='3d')
	ax.set_ylim(0,100)

	xs, ys, zs, colors = [], [], [], []
	base_xs, base_ys, base_zs, base_colors = [], [], [], []
	for x in range(0, 200, STRIDE):
		for y in range(0, 200, STRIDE):
			# color = np.array(list(cropped_image.getpixel((x, y)))) / 255.0
			base_xs.insert(0, x)
			base_ys.insert(0, 200-y)
			base_zs.insert(0, 0)
			# base_colors.insert(0, color)
			for z in np.argwhere(voxels[x][y] == 1).T[0]:
				xs.append(x)
				ys.append(200-y)
				zs.append(z)
				# colors.append(color)

	avg_z = sum(zs) / float(len(zs))
	zs = [max(z - avg_z, 0) for z in zs]

	xs = np.concatenate((base_xs, xs))
	ys = np.concatenate((base_ys, ys))
	zs = np.concatenate((base_zs, zs))
	colors = np.concatenate((base_colors, colors))

	ax.scatter(xs=xs, ys=zs, zs=ys, color=colors, s=8)

	if save:
		ax.view_init(elev=0, azim=90)
		plt.savefig("voxels_front.jpg")
		ax.view_init(elev=25, azim=50)
		plt.savefig("voxels_corner.jpg")
		ax.view_init(elev=0, azim=0)
		plt.savefig("voxels_side.jpg")
		ax.view_init(elev=0, azim=50)
		plt.savefig("voxels_side_front.jpg")

	else:
		ax.view_init(elev=0, azim=50)
		plt.show()


'''Visualizes the voxels on top of the original image.

Input:  The original image, the voxels, and the transform 
performed on the voxels during preprocessing.

Output: If save flag set, saves different views of the 
visualization in root. Otherwise, simply opens a window with
the visualization from a corner view.
'''
def visualize_voxels_original(image, voxels, transform, save=False):
	fig = plt.figure(figsize=(8,8))
	fig.subplots_adjust(top=1, bottom=0, left=0, right=1)

	ax = fig.add_subplot(111, projection='3d')

	###### Scaling Section #######
	x_scale=4.0
	y_scale=1.0
	z_scale=4.0

	scale=np.diag([x_scale, y_scale, z_scale, 1.0])
	scale=scale*(1.0/scale.max())
	scale[3,3]=1.0

	ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), scale)
	##############################

	Ainv = np.linalg.inv(transform['A'])
	b = transform['b']

	xs, ys, zs, colors = [], [], [], []
	base_xs, base_ys, base_zs, base_colors = [], [], [], []
	for x in range(0, 200, STRIDE):
		for y in range(0, 200, STRIDE):
			for z in np.argwhere(voxels[x][y] == 1).T[0]:
				inv_coords = Ainv.dot(np.subtract(np.array([x, y, z]), b.T)[0])
				xt, yt = int(inv_coords[0]), int(inv_coords[1])
				color = np.array(list(image.getpixel((xt, yt)))) / 255.0

				xs.append(xt)
				ys.append(image.size[1] - yt)
				zs.append(z)
				colors.append(color)

	# Shift z-coordinates of voxels to have 0 mean
	avg_z = sum(zs) / float(len(zs))
	# Clip negative z-coordinates to 0
	zs = [max(z - avg_z, 0) for z in zs]

	# Add base layer using original image
	for x in range(0, image.size[0], STRIDE):
		for y in range(0, image.size[1], STRIDE):
			color = np.array(list(image.getpixel((x, y)))) / 255.0
			xs.insert(0, x)
			ys.insert(0, image.size[1] - y)
			zs.insert(0, 0)
			colors.insert(0, color)

	ax.scatter(xs=xs, ys=zs, zs=ys, color=colors, s=5)
	ax.set_ylim(0,100)

	if save:
		# Used to crop whitespace
		bbox = fig.bbox_inches.from_bounds(0, 0, 6, 8)

		ax.view_init(elev=0, azim=90)
		plt.savefig("voxels_front.jpg")

		ax.view_init(elev=20, azim=40)
		plt.savefig("voxels_corner.jpg", bbox_inches=bbox)
		ax.view_init(elev=0, azim=0)
		plt.savefig("voxels_side.jpg", bbox_inches=bbox)
		ax.view_init(elev=0, azim=40)
		plt.savefig("voxels_side_front.jpg", bbox_inches=bbox)
	else:
		ax.view_init(elev=20, azim=40)
		plt.show()



if __name__ == '__main__':
    demo(save=True)
