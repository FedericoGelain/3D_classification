# 3D_classification
# 3D_classification
This implementation is based on a previous work <a href="https://github.com/FedericoGelain/3DP_homeworks/tree/main/HW4%20-%203D%20Deep%20Descriptors"> PointNet 3D deep descriptors </a>, which has been reorganized in Python files.
Install the project dependencies:
```sh
 pip install -r requirements.txt
```
Download the dataset <a href="https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset"> here </a>
First you need to create the voxel grids out of the downloaded meshes. To do so run the following command:
```sh
 python .\files_preparation.py ModelNet10 -f -x
```
which will fix the files headers, voxelize the meshes and save them in the <b>VoxelGrids</b> folder (or any other name specified)
To then run the training type the following command:
```sh
 python train_model.py
```

## 📁 Project Structure

```sh
└── PointNet/
  ├── requirements.txt
  ├── files_preparation.py
  ├── train_model.py
  ├── VoxelDataset.py
  └── voxelnet.py
```