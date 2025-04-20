import torch
import torch.nn as nn
import torch.nn.functional as F

# THe definition of each block follows the TinyPointNet paper

# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.conv  = nn.Conv1d(self.input_size, self.output_size, 1)
        self.bn    = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.conv(input)))

# Fully Connected with Batch Normalization
class FC_BN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size   = input_size
        self.output_size  = output_size
        self.lin  = nn.Linear(self.input_size, self.output_size)
        self.bn    = nn.BatchNorm1d(self.output_size)

    def forward(self, input):
        return F.relu(self.bn(self.lin(input)))

class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k

        self.mlp1 = MLP(self.k, 64)
        self.mlp2 = MLP(64, 128)
        self.mlp3 = MLP(128, 1024)

        self.fc_bn1 = FC_BN(1024, 512)
        self.fc_bn2 = FC_BN(512,256)

        self.fc3 = nn.Linear(256,k*k)


    def forward(self, input):
        # input.shape == (batch_size,n,3)

        bs = input.size(0)
        xb = self.mlp1(input)
        xb = self.mlp2(xb)
        xb = self.mlp3(xb)

        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)

        xb = self.fc_bn1(flat)
        xb = self.fc_bn2(xb)

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix
    
class TinyPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.input_transform = TNet(k=3) ## if you use the shot rotation matrix, this is not going to be used
        self.feature_transform = TNet(k=64)

        #first set of MLP layers to apply after the input transformation
        self.mlp1 = MLP(3, 64)
        self.mlp2 = MLP(64, 64)

        #second set of MLP layers to apply after the feature transformation
        self.mlp3 = MLP(64, 64)
        self.mlp4 = MLP(64, 128)
        self.mlp5 = MLP(128, 256)

    def forward(self, input):
        n_pts = input.size()[2]
        # matrix3x3 = self.input_transform(input)
        matrix3x3 = self.shot_canonical_rotation(input, 3)
        input_transform_output = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        a1 = self.mlp1(input_transform_output)
        a2 = self.mlp2(a1)

        feat_transform = self.feature_transform(a2)
        feature_transform_output = torch.bmm(torch.transpose(a2,1,2), feat_transform).transpose(1,2)

        a3 = self.mlp3(feature_transform_output)
        a4 = self.mlp4(a3)
        a5 = self.mlp5(a4)

        global_feature = torch.nn.MaxPool1d(a5.size(-1))(a5)

        return global_feature

    def shot_canonical_rotation(self, input, k):
        # input.shape == (batch_size, n, k)
        batch_size, n, _ = input.size()
        rotation_matrices = []

        for i in range(batch_size):
            features_batch = input[i]
            centroid = torch.mean(features_batch, dim=1, keepdim=True)
            centered_features = features_batch - centroid

            if torch.any(torch.isnan(centered_features)) or torch.any(torch.isinf(centered_features)):
                centered_features[torch.isnan(centered_features) | torch.isinf(centered_features)] = 1e-6

            distances = torch.norm(centered_features, dim=0)

            weights = 1.0 - distances
            weights[weights < 0] = 0

            weights_sum = torch.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum
            else:
                weights = torch.ones_like(weights) / n

            weighted_cov_matrix = torch.zeros((centered_features.size(0), centered_features.size(0)), device=input.device)

            ### compute the covariance matrix
            weighted_cov_matrix += torch.matmul(weights * centered_features, centered_features.t())

            weighted_cov_matrix += torch.eye(weighted_cov_matrix.size(-1), device=input.device) * 1e-6

            ### compute the eigenvectors of covariance matrix
            values, Vectors = torch.linalg.eig(weighted_cov_matrix)

            #remove the imaginary part (even if it's always equal to 0)
            Vectors = Vectors.type(torch.float32)
            values = values.type(torch.float32)

            #sort the eigenvectors in descending order based on the eigenvalues
            V = Vectors[:, torch.argsort(-values)]

            rotation_matrix = V[:, :k]

            for j in range(k):
                signs = torch.sign(torch.sum(centered_features * rotation_matrix[:, j].unsqueeze(1), dim=0))
                if torch.sum(signs >= 0) < torch.sum(signs < 0):
                    rotation_matrix[:, j] = -rotation_matrix[:, j]

            rotation_matrices.append(rotation_matrix)

        return torch.stack(rotation_matrices, dim=0)
