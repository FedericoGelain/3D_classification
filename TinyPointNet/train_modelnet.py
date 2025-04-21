import torch
from TinyPointNet import TinyPointNet
from torch.utils.data import DataLoader
from PointCloudDataset import PointCloudData
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def train(model, train_loader, valid_loader=None,  epochs=30, save=True):
    best_valid_loss = 1e10

    tinypointnetloss = torch.nn.TripletMarginLoss()

    # these lists keep track of the losses across epochs
    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        # local list of losses
        train_loss, valid_loss = [], []

        # train
        model.train()

        for (_, anchor, positive, negative, _, _, _, _) in tqdm(train_loader):

            # retrieve anchors, positives and negatives batch
            anchor   =   anchor.to(device).float().transpose(1,2)
            positive = positive.to(device).float().transpose(1,2)
            negative = negative.to(device).float().transpose(1,2)

            optimizer.zero_grad()

            # let PointNetTiny model compute the descriptors
            anchor_desc   = model(anchor)
            positive_desc = model(positive)
            negative_desc = model(negative)

            # compute the loss associated to these descriptors
            loss = tinypointnetloss(anchor_desc, positive_desc, negative_desc)

            # Backpropagate the gradient
            loss.backward()
            optimizer.step()

            # Keep track of the statistics
            train_loss.append(loss.item())
            #pbar.set_postfix(loss=curr_loss)

        train_loss = np.asarray(train_loss).mean()
        print(f'epoch {epoch} - train loss:', train_loss)
        train_losses.append(train_loss)

        # validation
        tinypointnet.eval()
        pbar = tqdm(valid_loader, leave=False)

        with torch.no_grad():
            for (_, anchor, positive, negative, _, _, _, _) in pbar:
                pbar.set_description(f"valid - epoch {epoch}")

                anchor   =   anchor.to(device).float().transpose(1,2)
                positive = positive.to(device).float().transpose(1,2)
                negative = negative.to(device).float().transpose(1,2)

                anchor_desc   = model(anchor)
                positive_desc = model(positive)
                negative_desc = model(negative)
                loss = tinypointnetloss(anchor_desc, positive_desc, negative_desc)
                curr_loss = loss.item()

                valid_loss.append(curr_loss)

                pbar.set_postfix(loss=curr_loss)

        valid_loss = np.asarray(valid_loss).mean()
        print(f'epoch {epoch} - valid loss:', valid_loss)
        valid_losses.append(valid_loss)

        # save the model
        if save and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            path = os.path.join("tinypointnetmodel.yml")
            print("best_valid_loss:", best_valid_loss, "saving model at", path)
            torch.save(model.state_dict(), path)

    return train_losses, valid_losses

def numpy_ewma_vectorized_v2(data, window=10):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

if __name__ == '__main__':
    train_ds  = PointCloudData(os.path.join("dataset", "train"), samples_per_epoch=500)
    valid_ds  = PointCloudData(os.path.join("dataset", "valid"), samples_per_epoch=500)
    test_ds   = PointCloudData(os.path.join("dataset", "test"),  samples_per_epoch=500, is_test_set=True)
    
    train_loader  = DataLoader( dataset=train_ds,  batch_size=50, shuffle=True  )
    valid_loader  = DataLoader( dataset=valid_ds,  batch_size=50, shuffle=False )
    test_loader   = DataLoader( dataset=test_ds,   batch_size=20, shuffle=False )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    tinypointnet = TinyPointNet()
    tinypointnet.to(device)

    optimizer = torch.optim.Adam(tinypointnet.parameters(), lr=0.0005)

    train_losses, valid_losses = train(tinypointnet, train_loader, valid_loader, save=True)

    plt.plot(train_losses, label="train losses", color='r', alpha=0.2)
    plt.plot(valid_losses, label="valid losses", color='b', alpha=0.2)
    plt.plot(numpy_ewma_vectorized_v2(np.array(train_losses)), label="train losses EMA", color='r')
    plt.plot(numpy_ewma_vectorized_v2(np.array(valid_losses)), label="valid losses EMA", color='b')
    plt.legend()

    plt.show()

    path = os.path.join("tinypointnetmodel.yml")
    tinypointnet = TinyPointNet()
    tinypointnet.load_state_dict(torch.load(path))
    tinypointnet.to(device)

    test_ds.generate_test_set(0, 2000)

    ## build the ground truth nearest neighbors
    ## in other words, find thepoints in the noisy test points
    ## that are the nearest neighbors to the original, sampled, test points
    test_ds.generate_noisy_test_set(0)

    descs   = test_ds.compute_descriptors(tinypointnet, device)
    descs_n = test_ds.compute_descriptors(tinypointnet, device, noisy=True)

    correct = tot = 0

    for row in tqdm(range(test_ds.test_points_sampled.shape[0])):
        desc = descs[row, :]
        dists = []
        anchor       = test_ds.test_points_sampled[row]
        true_near_pt = test_ds.test_points_sampled_n[row]

        for row2 in range(test_ds.test_points_sampled_n.shape[0]):
            desc2 = descs_n[row2, :]
            dist = np.linalg.norm(desc - desc2)
            dists.append(dist)

        min_row = np.argmin(np.asarray(dists))

        pred_pt = test_ds.test_points_sampled_n[min_row].squeeze()

        dist = np.linalg.norm(true_near_pt - pred_pt)

        if dist<test_ds.radius:
            correct += 1
        tot += 1

    print()
    print(f"accuracy: {correct*100/tot:6.3f}%")




