from manager import get_batch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as im

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
	images, vertex_lst, voxel_lst = get_batch(1)
	vertices = vertex_lst[0]
	voxels = voxel_lst[0]
	image = images[0]


	plt.figure()
	plt.imshow(image)
	plt.plot(subsample(vertices[0], 30), subsample(vertices[1], 30), 'g.')
	plt.show()


	# picture = Image.fromarray(image, 'RGB')
	# picture.show()


demo()
