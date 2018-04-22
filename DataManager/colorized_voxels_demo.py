from manager import get_batch
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np

STRIDE = 1 # Higher stride means more subsampling (faster rendering, worse quality)

########################
### Helper Functions ###
########################

def subsample(vertices, rate):
        """ Returns a subset of the vertices given. Used for plotting transformed meshes
        over cropped images in the section 'Plotting Examples'.

        Adjust the subsampling rate to plot sparser/thicker meshes on your image.

        THIS FUNCTION WAS COPIED OVER FROM PREPROCESS.PY AND IS FOR DEMONSTRATION PURPOSES
        ONLY. IT SHOULD BE DELETED ONCE THE MAIN FUNCTIONALITY OF THIS FILE IS IMPLEMENTED.
        """
        return [vertices[i] for i in range(len(vertices)) if i%rate == 0]


"""Displays a single image retrieved using get_batch()."""
def demo():
	images, voxel_lst = get_batch(1)
	image = Image.fromarray(images[0], 'RGB')
	voxels = voxel_lst[0]

	visualize_voxels(image, voxels)


def visualize_voxels(image, voxels):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs, ys, zs, colors = [], [], [], []
	for x in range(0, 200, STRIDE):
		for y in range(0, 200, STRIDE):
			for z in np.argwhere(voxels[x][y] == 1).T[0]:
				xs.append(x)
				ys.append(y)
				zs.append(z)
				colors.append(np.array(list(image.getpixel((x, y)))) / 255.0)

	ax.scatter(xs=xs, ys=ys, zs=zs, color=colors, s=5)
	plt.show()


demo()
