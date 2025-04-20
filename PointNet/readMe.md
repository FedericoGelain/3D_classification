# 3D_classification
This implementation is based on a previous work <a href="https://github.com/FedericoGelain/3DP_homeworks/tree/main/HW4%20-%203D%20Deep%20Descriptors"> PointNet 3D deep descriptors </a>, which has been reorganized in Python files.
Install the project dependencies:
```sh
 pip install -r requirements.txt
```
If not already installed, make sure to also add glob:
```sh
 pip install glob2
```
Download the dataset <a href="https://drive.google.com/drive/folders/1IweJGcOeOZN3wY79i2jFt3JE1bd7G51Z?usp=share_link"> here </a> and put the 3 downloaded folders inside one called <b>dataset</b> <br />
To run the training type the following command:
```sh
 python train_modelnet.py
```

## 📁 Project Structure (showing how the dataset folder should be placed)

```sh
└── PointNet/
  ├── requirements.txt
  ├── dataset
  │   ├── train
  │   ├── valid
  │   └── test
  │ 
  ├── PointCloudDataset.py
  ├── TinyPointNet.py
  ├── train_modelnet.py
  └── tinypointnetmodel.yml
```