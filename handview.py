#!/home/postgres/mp_env/bin/python

import os
import glob
import trimesh
import random
import numpy as np
from matplotlib import pyplot as plt

#mesh_file='/home/postgres/workspace/project/data/off/Call/good1.jpg_1.off'
mesh_file='/home/postgres/workspace/project/data/off/20230322_30/Fist/Fist_253.jpg_9.off'
mesh = trimesh.load(mesh_file)
#mesh.show()

points = mesh.vertices
print('>>points shape', points.shape)


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
#ax.set_axis_off()
plt.show()
