import os
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from scipy.ndimage import rotate
import torch

def voxelgrid_to_dense_tensor(voxel_grid, resolution=(32, 32, 32)):
    voxels = np.zeros(resolution, dtype=np.float32)

    for voxel in voxel_grid.get_voxels():
        x, y, z = voxel.grid_index
        if 0 <= x < resolution[0] and 0 <= y < resolution[1] and 0 <= z < resolution[2]:
            voxels[x, y, z] = 1.0

    return torch.tensor(voxels).unsqueeze(0)  # Shape: (1, D, H, W)

class VDataset(Dataset):
    # constructor essentially here you pass all the items that will be added in the dataset (train or test items with labels)
    def __init__(self, root_dir, mode="train", transform=None):
        if(mode != "train" and mode != "test"):
            raise ValueError(f"Invalid mode: {mode}. Must be one either 'train' or 'test'.")
        
        # Collect all folders names and files
        self.classes = [dir for dir in sorted(os.listdir(root_dir))]
        self.classes_encoding = {self.classes[i]: i for i in range(len(self.classes))}

        # Get the list of all items path and corresponding labels
        self.items = []

        for obj_class in self.classes:
            folder_path = os.path.join(root_dir, obj_class, mode)

            for filename in os.listdir(folder_path):
                obj = {}
                obj["path"] = os.path.join(folder_path, filename)
                obj["label"] = self.classes_encoding[obj_class]

                self.items.append(obj)

        self.transform = transform

    def __len__(self):
        return len(self.items)
    
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        voxel_grid = o3d.io.read_voxel_grid(self.items[idx]["path"])
        label = self.items[idx]["label"]

        # Convert to dense voxel tensor
        x = voxelgrid_to_dense_tensor(voxel_grid)

        if self.transform:
            x = self.transform(x)  # apply any PyTorch-compatible transforms here

        return x, label


def voxelgrid_to_numpy(voxel_grid, resolution=64):
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel_size = voxel_grid.voxel_size
    origin = voxel_grid.origin

    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index  # (x, y, z) index
        if all(0 <= i < resolution for i in idx):
            grid[idx[0], idx[1], idx[2]] = 1.0

    return grid


def numpy_to_voxelgrid(voxel_array, voxel_size=1.0, origin=[0, 0, 0]):
    # Get non-zero voxel coordinates
    indices = np.argwhere(voxel_array > 0.5)  # Only keep "occupied" voxels
    indices = indices.astype(np.float32) * voxel_size  # scale by voxel size

    # Convert to Open3D PointCloud then to VoxelGrid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(indices + origin)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size,
        min_bound=origin,
        max_bound=(np.array(voxel_array.shape) * voxel_size + origin)
    )

    return voxel_grid

if __name__ == '__main__':
    root = "Voxels"
    classes = [dir for dir in sorted(os.listdir(root))]
    classes_encoding = {classes[i]: i for i in range(len(classes))}

    pcd = o3d.io.read_voxel_grid('Voxels\\toilet\\train\\toilet_0106.ply') # Read the point cloud
    o3d.visualization.draw_geometries([pcd], width=600, height=400)

    # Convert to dense voxel array
    voxels = voxelgrid_to_numpy(pcd)

    # axes=(1, 2) → rotate around Z
    # axes=(0, 2) → rotate around Y
    # axes=(0, 1) → rotate around X
    rotated_voxels = rotate(voxels, angle=45, axes=(1, 2), reshape=False, order=1)

    rotated_voxel_grid = numpy_to_voxelgrid(rotated_voxels)
    o3d.visualization.draw_geometries([rotated_voxel_grid], width=600, height=400) 

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    #np_array = np.asarray(pcd.points)

    
    #rotated_grids = []
    #for axes in rotation_axes:
    #    for i in range(3):
    #        rotated_grid = np.rot90(voxel_grid, i, axes=axes).copy()
    #        rotated_grids.append(rotated_grid)
    #return rotated_grids
