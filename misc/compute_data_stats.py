# for each dataset, compute the mean and std of each of the keys. The data is stored in h5 files.

import os
import h5py
import numpy as np

import argparse
import json
import glob

data_path = r"C:\Users\20195435\Documents\TUe\Tasti\Solvey\code\BSMS-GNN\data\plate\outputs_train"


normalization_info = {}
normalization_info['world_pos'] = {}
normalization_info['world_pos']['all'] = []
normalization_info['stress'] = {}
normalization_info['stress']['all'] = []
normalization_info['mesh_pos'] = {}
normalization_info['mesh_pos']['all'] = []


# example = r"/app/code/BSMS-GNN/data/plate/outputs_train/0.h5"
# with h5py.File(example, 'r') as f:
#     print(f.keys())


# loop through all the files in the directory
for filename in os.listdir(data_path):
    if filename.endswith('.h5'):
        print(f"Processing {filename}...")
        with h5py.File(os.path.join(data_path, filename), 'r') as f:
            # print(f.keys())
            world_pos = f['world_pos'][()]
            # print(world_pos.shape)
            # flatten the world_pos array to 1D
            world_pos = world_pos.reshape(-1, world_pos.shape[-1])
            stress = f['stress'][()]
            # print(stress.shape)
            # print(stress)
            # flatten the stress array to 1D
            stress = stress.reshape(-1, stress.shape[-1])
            mesh_pos = f['mesh_pos'][()]
            # flatten the mesh_pos array to 1D
            mesh_pos = mesh_pos.reshape(-1, mesh_pos.shape[-1])
            # print(world_pos.shape, stress.shape, mesh_pos.shape)
            normalization_info['world_pos']['all'].append(world_pos)
            normalization_info['stress']['all'].append(stress)
            normalization_info['mesh_pos']['all'].append(mesh_pos)
            # print(world_pos.shape)

# compute the mean and std of each of the keys
normalization_info['world_pos']['mean'] = np.mean(np.concatenate(normalization_info['world_pos']['all'], axis=0), axis=0).tolist()
normalization_info['world_pos']['std'] = np.std(np.concatenate(normalization_info['world_pos']['all'], axis=0), axis=0).tolist() 
normalization_info['stress']['mean'] = np.mean(np.concatenate(normalization_info['stress']['all'], axis=0), axis=0).tolist()
normalization_info['stress']['std'] = np.std(np.concatenate(normalization_info['stress']['all'], axis=0), axis=0).tolist()
normalization_info['mesh_pos']['mean'] = np.mean(np.concatenate(normalization_info['mesh_pos']['all'], axis=0), axis=0).tolist()
normalization_info['mesh_pos']['std'] = np.std(np.concatenate(normalization_info['mesh_pos']['all'], axis=0), axis=0).tolist()

print("world_pos mean: ", normalization_info['world_pos']['mean'])
print("world_pos std: ", normalization_info['world_pos']['std'])
print("stress mean: ", normalization_info['stress']['mean'])
print("stress std: ", normalization_info['stress']['std'])
print("mesh_pos mean: ", normalization_info['mesh_pos']['mean'])
print("mesh_pos std: ", normalization_info['mesh_pos']['std'])

# remove the all key from the normalization_info dictionary
del normalization_info['world_pos']['all']
del normalization_info['stress']['all']
del normalization_info['mesh_pos']['all']

# write the normalization info to a json file
with open(os.path.join(data_path, 'normalization_info.json'), 'w') as f:
    # make sure it is json serializable
    normalization_info['world_pos']['mean'] = [float(i) for i in normalization_info['world_pos']['mean']]
    normalization_info['world_pos']['std'] = [float(i) for i in normalization_info['world_pos']['std']] 
    normalization_info['stress']['mean'] = [float(i) for i in normalization_info['stress']['mean']]
    normalization_info['stress']['std'] = [float(i) for i in normalization_info['stress']['std']]
    normalization_info['mesh_pos']['mean'] = [float(i) for i in normalization_info['mesh_pos']['mean']]
    normalization_info['mesh_pos']['std'] = [float(i) for i in normalization_info['mesh_pos']['std']]
    # write the normalization info to a json file
    json.dump(normalization_info, f, indent=4)