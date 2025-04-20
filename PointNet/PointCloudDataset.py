import numpy as np
import copy
import glob

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# pyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# a nice training progress bar
from tqdm import tqdm
import open3d as o3d

class PointCloudData(Dataset):
    def __init__(self,
                dataset_path: str,
                samples_per_epoch: int,
                points_to_sample:int = 200000,
                radius:float =0.02,
                min_dist=1.5e-2,
                N = 750,
                noise_mean=0,
                noise_variance = 6e-5,
                is_test_set=False,
                ):
        """
          INPUT
              dataset_path: path to the dataset folder
              transform   : transform function to apply to point cloud
        """

        self.radius = radius
        self.min_dist = min_dist
        self.N = N
        self.samples_per_epoch = samples_per_epoch
        self.points_to_sample = points_to_sample
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.is_test_set = is_test_set

        # _n means noised version
        self.mesh = []
        self.pcds, self.pcds_n = [], []
        self.KDtrees, self.KDtrees_n = [], []

        # if it's the test set, pre-define a random rotation matrix
        if self.is_test_set:
            self.common_rot_mat = self.get_xyz_random_rotation()

        for file in glob.glob(dataset_path + "/*.ply"):
            print("parsing file", file)
            mesh = o3d.io.read_triangle_mesh(file)
            pcd = mesh.sample_points_uniformly(self.points_to_sample)
            if self.is_test_set:
                pcd.rotate(self.common_rot_mat.as_matrix(), center=(0, 0, 0))
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)

            pcd_n = self.apply_noise(mesh.sample_points_uniformly(self.points_to_sample), self.noise_mean, self.noise_variance)
            if self.is_test_set:
                pcd_n.rotate(self.common_rot_mat.as_matrix(), center=(0, 0, 0))
            pcd_n_tree = o3d.geometry.KDTreeFlann(pcd_n)

            self.mesh.append(mesh)

            self.pcds.append(np.asarray(pcd.points))
            self.pcds_n.append(np.asarray(pcd_n.points))

            self.KDtrees.append(pcd_tree)
            self.KDtrees_n.append(pcd_n_tree)

    # function to apply noise
    def apply_noise(self, pcd, mu, sigma):
        noisy_pcd = copy.deepcopy(pcd)
        points = np.asarray(noisy_pcd.points)
        points += np.random.normal(mu, sigma, size=points.shape)
        noisy_pcd.points = o3d.utility.Vector3dVector(points)
        return noisy_pcd

    def __len__(self):
        return self.samples_per_epoch

    def get_xyz_random_rotation(self):
        random_rotation_on_xyz_axis = np.random.rand(3) * 2 * np.pi
        return R.from_euler('xyz', random_rotation_on_xyz_axis, degrees=False)

    def get_point_cloud_center(self, pc_points):
        return pc_points.mean(axis=0)

    def apply_rotation(self, point, rot_mat, pcd_center):
        return np.dot(point.reshape(1, 3), rot_mat.as_matrix().T)[0, :]

    def apply_rotation_pc(slef, points, rot_mat, pcd_center):
        return np.dot(points, rot_mat.as_matrix().T)

    def bufferize_pointcloud(self, points, N):
        pc = np.zeros((self.N, 3), dtype=np.float32)
        pc[:min(self.N, points.shape[0]), :] = points[:min(self.N, points.shape[0]), :]
        return pc

    def sample_anchor_point(self, pcd_points, pcd_tree):
        #sample the anchor point
        anchor_idx = np.random.randint(0, len(pcd_points))
        anchor_pt = pcd_points[anchor_idx]

        #retrieve the neighbours of the anchor point (the ones whose distance is smaller than the set radius)
        _, anchor_neighborhood_idxs, _ = pcd_tree.search_radius_vector_3d(anchor_pt, self.radius)

        return anchor_pt, anchor_neighborhood_idxs

    def sample_positive_point(self, pcd_n_points, pcd_n_tree, anchor_pt):
        _, noisy_anchor_nn_idx, _ = pcd_n_tree.search_knn_vector_3d(anchor_pt, 1)

        #retrieve the point in the noisy point cloud that's the nearest to the
        #anchor point, which will be used as the positive sample
        pos_pt = pcd_n_points[noisy_anchor_nn_idx[0]]

        #retrieve the neighbours of the positive point (the ones whose distance is smaller than the set radius)
        _, noisy_positive_neighborhood_idxs, _ = pcd_n_tree.search_radius_vector_3d(pos_pt, self.radius)

        return pos_pt, noisy_positive_neighborhood_idxs

    def sample_negative_point(self, pcd_n_points, pcd_n_tree, anchor_pt):
        while True:
          #sample a point uniformly at random
          neg_idx = np.random.randint(0, len(pcd_n_points))
          neg_pt = pcd_n_points[neg_idx]

          #if the distance between sampled point and anchor point is greater than the set minimum distance
          #then the negative sample has been found and the loop can finish
          if np.linalg.norm(anchor_pt - neg_pt) >= self.min_dist:
            break

        #retrieve the neighbours of the negative point
        _, noisy_negative_neighborhood_idxs, _ = pcd_n_tree.search_radius_vector_3d(neg_pt, self.radius)

        return neg_pt, noisy_negative_neighborhood_idxs

    def get_sampled_pointcloud(self, mesh_idx, N):
        idxs = np.arange(0, self.pcds[mesh_idx].shape[0], 1)
        np.random.shuffle(idxs)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcds[mesh_idx][idxs[:N]])
        return pcd

    def generate_test_set(self, mesh_idx, N=500):

        self.test_points = self.pcds[mesh_idx]

        # sample test points randomly
        idxs = np.arange(0, self.test_points.shape[0], 1)
        np.random.shuffle(idxs)
        #self.test_points_sampled = self.test_points[idxs[:N]]

        # keypoints sampling directly using open3d functions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.pcds[mesh_idx])
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                         salient_radius=0.005,
                                                         non_max_radius=0.005,
                                                         gamma_21=0.4,
                                                         gamma_32=0.5)
        self.test_points_sampled = np.random.permutation(np.asarray(keypoints.points))

        self.test_tree = self.KDtrees[mesh_idx]

    def generate_noisy_test_set(self, mesh_idx):
        self.test_points_n = self.pcds_n[mesh_idx]
        self.test_tree_n = self.KDtrees_n[mesh_idx]

        test_points_sampled_n = []
        for i in tqdm(range(self.test_points_sampled.shape[0])):
            _, gt_nn_idx, _ = self.test_tree_n.search_knn_vector_3d(self.test_points_sampled[i], 1)
            gt_nearest_point = self.test_points_n[gt_nn_idx].squeeze()
            test_points_sampled_n.append(gt_nearest_point)

        self.test_points_sampled_n = np.asarray(test_points_sampled_n)

    def compute_descriptors(self, tinypointnet, device, noisy=False):
        tinypointnet.eval()

        if noisy:
            queries = self.test_points_sampled_n
            tree    = self.test_tree_n
            points  = self.test_points_n
        else:
            queries = self.test_points_sampled
            tree    = self.test_tree
            points  = self.test_points
        tot_queries = queries.shape[0]

        descriptors = np.zeros((tot_queries, 256), dtype=np.float32)
        for i in tqdm(range(tot_queries)):
            pt = queries[i]

            # find neighborhood of points
            _, idx, _ = tree.search_radius_vector_3d(pt, self.radius)
            point_set = points[idx]

            # normalize the points
            point_set = (point_set - pt)

            pc = np.zeros((self.N, 3), dtype=np.float32)
            pc[:min(self.N, point_set.shape[0]), :] = point_set[:min(self.N, point_set.shape[0]), :]

            # transform
            anchor  = torch.from_numpy(pc).unsqueeze(0).float().transpose(1,2).to(device)
            descriptors[i, :] = tinypointnet(anchor)[0, :, 0].cpu().detach().numpy()
        return descriptors

    def __getitem__(self, _):
        mesh_idx = np.random.randint(0, len(self.mesh))

        pcd_points = self.pcds[mesh_idx]        # anchor will be drawn from this
        pcd_n_points = self.pcds_n[mesh_idx]    # positive and negative will be drawn from this

        # ANCHOR: select a random anchor point
        anchor_pt, anchor_neighborhood_idxs = self.sample_anchor_point(pcd_points, self.KDtrees[mesh_idx])

        # POSITIVE: find corresponding point in the noisy point cloud
        pos_pt, noisy_positive_neighborhood_idxs = self.sample_positive_point(pcd_n_points, self.KDtrees_n[mesh_idx], anchor_pt)

        # NEGATIVE: find far point (at least at distance min_dist)
        neg_pt, noisy_negative_neighborhood_idxs = self.sample_negative_point(pcd_n_points, self.KDtrees_n[mesh_idx], anchor_pt)
        if neg_pt is None: # it should never fail, but if it fails: restart experiment
            quit("FAIL: restart experiment")


        # get points
        point_set_anchor   = pcd_points[anchor_neighborhood_idxs]
        point_set_positive = pcd_n_points[noisy_positive_neighborhood_idxs]
        point_set_negative = pcd_n_points[noisy_negative_neighborhood_idxs]

        if not self.is_test_set:
            # generate a random rotation
            rot_mat = self.get_xyz_random_rotation()

            # apply the random rotation to point cloud and points
            pcd_points_center = self.get_point_cloud_center(pcd_points)
            pcd_n_points_center = self.get_point_cloud_center(pcd_n_points)
            anchor_pt = self.apply_rotation(anchor_pt, rot_mat, pcd_points_center)
            pos_pt    = self.apply_rotation(pos_pt, rot_mat, pcd_n_points_center)
            neg_pt    = self.apply_rotation(neg_pt, rot_mat, pcd_n_points_center)
            point_set_anchor   = self.apply_rotation_pc(point_set_anchor, rot_mat, pcd_points_center)
            point_set_positive = self.apply_rotation_pc(point_set_positive, rot_mat, pcd_n_points_center)
            point_set_negative = self.apply_rotation_pc(point_set_negative, rot_mat, pcd_n_points_center)
        else:
            rot_mat = self.common_rot_mat

        # center points around their centroid
        point_set_anchor   = (point_set_anchor - anchor_pt)
        point_set_positive = (point_set_positive - pos_pt)
        point_set_negative = (point_set_negative - neg_pt)

        # copy points coordinates to a fixed dimension np.array
        point_set_anchor   = self.bufferize_pointcloud(point_set_anchor  , self.N)
        point_set_positive = self.bufferize_pointcloud(point_set_positive, self.N)
        point_set_negative = self.bufferize_pointcloud(point_set_negative, self.N)

        # transform from numpy to torch.Tensor
        point_set_anchor    = torch.from_numpy(point_set_anchor)
        point_set_positive  = torch.from_numpy(point_set_positive)
        point_set_negative  = torch.from_numpy(point_set_negative)

        return mesh_idx, point_set_anchor, point_set_positive, point_set_negative, anchor_pt, pos_pt, neg_pt, rot_mat.as_matrix()