import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


class Visualize3D:
    def __init__(self):
        plt.interactive(True)
        fig = plt.figure()
        self.ax = plt.axes(projection="3d")
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([0, 5])
        self.ax.set_xlabel("X Axis")
        self.ax.set_ylabel("Y Axis")
        self.ax.set_zlabel("Z Axis")

    def update(self, poses):
        X, Y, Z, U, V = [[] for _ in range(5)]
        for pose in poses:
            pose = np.array(poses[pose][0])
            trans = pose[:3, 3]
            rot = pose[:3, :3]
            rot = Rotation.from_matrix(rot).as_euler("xyz")
            yaw = rot[2]
            new_x = math.sin(yaw)
            new_y = math.cos(yaw)
            X.append(trans[0])
            Y.append(trans[1])
            Z.append(trans[2])
            U.append(new_x)
            V.append(new_y)

        self.ax.quiver(X, Y, Z, U, V, 0)
        plt.draw()
        plt.pause(0.02)
        self.ax.cla()
        plt.show()
