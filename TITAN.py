import os
import argparse
import h5py
import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm
import time

# Load model once
model = AutoModel.from_pretrained(
    "MahmoodLab/TITAN",
    trust_remote_code=True
)



# TODO check!

# for TCGA i get patch_size 550 
# but for TracerX i get patch_size 290 

# This peram must be correct... othermise may make some extra batch effect? 

### Must take conch 1.5 features and coords and then extract the slide embedding... 

# TODO: not hardcode the dist value  

def extract_slide_embedding(h5_path):
    with h5py.File(h5_path, 'r') as file:
        features = file['features'][:][None, ...]
        coords = file['coords'][:][None, ...].astype(np.int64)
        features = torch.from_numpy(features)
        coords = torch.from_numpy(coords)
        patch_size_lv0 = 290
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_lv0)
    return slide_embedding.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='Extract slide embeddings from .h5 files in a directory.')
    parser.add_argument('--input_dir', required=True, help='Directory containing input .h5 files')
    parser.add_argument('--output_dir', required=True, help='Directory to save output embedding .h5 files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    h5_files = [fname for fname in os.listdir(args.input_dir) if fname.endswith('.h5')]
    for fname in tqdm(h5_files, desc='Processing files'):
        in_path = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        start_time = time.time()
        try:
            embedding = extract_slide_embedding(in_path)
            with h5py.File(out_path, 'w') as f:
                f.create_dataset('embedding', data=embedding)
            elapsed = time.time() - start_time
            tqdm.write(f'Saved embedding to {out_path} (Time: {elapsed:.2f}s)')
        except Exception as e:
            tqdm.write(f'Error processing {in_path}: {e}')

if __name__ == '__main__':
    main() 