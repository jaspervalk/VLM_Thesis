import pickle
import os

path = './features/things/torchvision/resnet50/penultimate/features.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

print(f"Top-level keys: {list(data.keys())}")

if 'filenames' in data:
    print(f"filenames[0:5]: {data['filenames'][0:5]}")
    print(f"filenames type: {type(data['filenames'][0])}")
if 'penultimate' in data:
    print(f"penultimate shape: {data['penultimate'].shape}")
