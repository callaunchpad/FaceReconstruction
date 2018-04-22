from manager import get_batch
from PIL import Image


"""Displays a single image retrieved using get_batch()."""
def demo():
	images, vertex_lst, voxel_lst = get_batch(1)
	voxels = voxel_lst[0]
	image = images[0]


	picture = Image.fromarray(image, 'RGB')
	picture.show()

demo()