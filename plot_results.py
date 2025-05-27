import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import h5py
import os

def plot_tensor_sample(data, prediction, sample_idx, cmap='viridis', s=20):
    """
    Plot a single sample from a tensor of shape (400, n, 4) as a 3D scatter,
    coloring by the 4th dimension.

    Parameters
    ----------
    data : np.ndarray or torch.Tensor
        Input tensor/array of shape (400, n, 4).
    sample_idx : int
        Index of the sample to plot (0 <= sample_idx < 400).
    cmap : str, optional
        Name of the matplotlib colormap to use for coloring (default 'viridis').
    s : float, optional
        Marker size for scatter points (default 20).

    Raises
    ------
    ValueError
        If sample_idx is out of range or data does not have shape (400, n, 4).
    """
    # Convert torch.Tensor to numpy array if needed
    if isinstance(data, torch.Tensor):
        # ensure tensor is on CPU and detached
        data = data.detach().cpu().numpy()
        prediction = prediction.detach().cpu().numpy()

    # Extract the sample
    sample = data[sample_idx]  # shape: (n, 4)
    x, y, z, c = sample[:,0], sample[:,1], sample[:,2], sample[:,3]
    # Extract the prediction
    pred = prediction[sample_idx]  # shape: (n, 4)
    pred_x, pred_y, pred_z, pred_c = pred[:,0], pred[:,1], pred[:,2], pred[:,3]

    # Create 3D scatter plot of the samples next to each other
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s)
    ax.set_title(f'Sample {sample_idx} - Original')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.colorbar(sc, ax=ax, label='Color Scale')


    # Create 3D scatter plot of the predictions
    ax2 = fig.add_subplot(122, projection='3d')
    sc2 = ax2.scatter(pred_x, pred_y, pred_z, c=pred_c, cmap=cmap, s=s)
    ax2.set_title(f'Sample {sample_idx} - Prediction')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')
    plt.colorbar(sc2, ax=ax2, label='Color Scale')
    plt.tight_layout()
    # plt.show()
    # save the figure
    save_path = os.path.join(os.getcwd(), f'sample_{sample_idx}.png')
    plt.savefig(save_path)
    


if __name__ == '__main__':
    # Example usage:
    # Create dummy data: 400 samples, each with 1000 points in 4D
    # Here we'll use random numbers for demonstration.

    # read file h5
    for index in range(200):
    # index = 0
        path = r"C:\Users\20195435\Documents\TUe\Tasti\Solvey\code\BSMS-GNN\data\plate\outputs_test\0.h5"
        with h5py.File(path, 'r') as f:
            print(list(f.keys()))
            pos = f['world_pos'][:]
            stress = f['stress'][:]
            print(pos.shape, stress.shape)
            data = np.concatenate((pos, stress), axis=2)

        # plot_tensor_sample(data, sample_idx=index)

        # Or if you prefer PyTorch:
        # dummy = torch.rand(400, 1000, 4)
        path = r"C:\Users\20195435\Documents\TUe\Tasti\Solvey\code\BSMS-GNN\res\cool-lion-56\ours\rollout_RMSE_epoch_20\0.h5"
        with h5py.File(path, 'r') as f:
            print(list(f.keys()))
            predicton = f['predictions'][:]
        print(predicton.shape)  # Check the shape of the data
        # Plot sample #42
        plot_tensor_sample(data, predicton, sample_idx=index)
