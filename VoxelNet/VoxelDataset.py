import os
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
from scipy.ndimage import rotate
import torch
import random

def voxelgrid_to_dense_tensor(voxel_grid, resolution=(32, 32, 32)):
    voxels = np.zeros(resolution, dtype=np.float32)

    for voxel in voxel_grid.get_voxels():
        x, y, z = voxel.grid_index
        if 0 <= x < resolution[0] and 0 <= y < resolution[1] and 0 <= z < resolution[2]:
            voxels[x, y, z] = 1.0

    return torch.tensor(voxels).unsqueeze(0)  # Shape: (1, D, H, W)

class VoxelAugmentation:
    def __init__(self, rotation_prob=0.8, flip_prob=0.3):
        self.rotation_prob = rotation_prob
        self.flip_prob = flip_prob
    
    def __call__(self, x):
        # Random 90-degree rotations
        if random.random() < self.rotation_prob:
            k = random.randint(0, 3)
            axis = random.choice([(1, 2), (1, 3), (2, 3)])
            x = torch.rot90(x, k, axis)
        
        # Random flips
        if random.random() < self.flip_prob:
            axis = random.choice([1, 2, 3])
            x = torch.flip(x, [axis])
        
        return x
    
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

if __name__ == '__main__':
    print('Test')
