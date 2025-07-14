import numpy as np
import time
import os
import argparse
import pdb
from functools import partial
import random

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import tiffslide
from tqdm import tqdm


from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_random_sample(batch, coords, slide_id, feat_dir, model_name, sample_idx):
	"""
	Save a single random sample image for visual inspection.
	
	Args:
		batch: Tensor containing a single image (batch size 1)
		coords: Coordinates of the patch
		slide_id: Name of the slide
		feat_dir: Base feature directory
		model_name: Name of the model to determine correct denormalization
		sample_idx: Index of this sample for filename
	"""
	samples_dir = os.path.join(feat_dir, 'samples', slide_id)
	
	# Define normalization values for different models
	if model_name in ['conch_v1']:
		# OPENAI normalization values
		mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
		std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
	elif model_name in ['H-optimus-0']:
		# H-optimus specific normalization
		mean = torch.tensor([0.707223, 0.578729, 0.703617]).view(3, 1, 1)
		std = torch.tensor([0.211883, 0.230117, 0.177517]).view(3, 1, 1)
	else:
		# Default ImageNet normalization (used by most models)
		mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
		std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
	
	# Convert tensor to PIL Image for saving
	img_tensor = batch[0]  # Get the single image from batch
	
	# Denormalize
	img_tensor = img_tensor * std + mean
	img_tensor = torch.clamp(img_tensor, 0, 1)
	
	# Convert to PIL Image
	img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
	img_pil = Image.fromarray(img_np)
	
	# Create filename with coordinates for reference
	coord = coords[0]
	filename = f"random_patch_{sample_idx:03d}_coord_{coord[0]}_{coord[1]}.jpg"
	filepath = os.path.join(samples_dir, filename)
	
	# Save the image
	img_pil.save(filepath, quality=95)
	
	if sample_idx == 0:  # Print only for first sample to avoid spam
		print(f"Saving random samples to {samples_dir}")

def compute_w_loader(output_path, loader, model, device, model_name, slide_id, feat_dir, save_samples=10, verbose=1):
	"""
	Args:
		output_path: directory to save computed features (.h5 file)
		loader: DataLoader object
		model: pytorch model
		device: torch device
		model_name: name of the model being used
		slide_id: name of the slide for sample saving
		feat_dir: base feature directory for sample saving
		save_samples: number of sample images to save (0 to disable)
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'Processing a total of {len(loader)} batches')

	mode = 'w'
	model.eval()  # Ensure the model is in evaluation mode
	
	# Setup for random sampling across batches
	saved_samples = 0
	total_batches = len(loader)
	if save_samples > 0 and total_batches > 0:
		# Calculate sampling probability to get approximately save_samples images
		# We'll sample roughly 1 image per batch until we have enough samples
		samples_per_batch = max(1, save_samples // total_batches)
		sample_probability = min(1.0, save_samples / total_batches)
		samples_dir = os.path.join(feat_dir, 'samples', slide_id)
		os.makedirs(samples_dir, exist_ok=True)
	
	for count, data in enumerate(tqdm(loader)):
		try:
			if verbose > 0:
				print(f"Processing batch {count + 1}/{len(loader)}")
			
			with torch.inference_mode():
				batch = data['img']
				coords = data['coord'].numpy().astype(np.int32)
				
				# Randomly sample images from across all batches
				if save_samples > 0 and saved_samples < save_samples:
					# Decide whether to sample from this batch
					if random.random() < sample_probability or saved_samples == 0:
						# Randomly select one image from this batch
						batch_size = batch.shape[0]
						random_idx = random.randint(0, batch_size - 1)
						
						sample_img = batch[random_idx:random_idx+1]  # Keep batch dimension
						sample_coord = coords[random_idx:random_idx+1]
						
						save_random_sample(sample_img, sample_coord, slide_id, feat_dir, model_name, saved_samples)
						saved_samples += 1

				batch = batch.to(device, non_blocking=True)
				features = model(batch)
				
				if model_name == "virchow":
					class_token = features[:, 0]    # size: n x 1280
					patch_tokens = features[:, 1:]  # size: n x 256 x 1280
					# concatenate class token and average pool of patch tokens
					features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: n x 2560

				elif model_name == "virchow_v2":
					class_token = features[:, 0]    # size: 1 x 1280
					patch_tokens = features[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those

					# concatenate class token and average pool of patch tokens
					features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

				else:
					print(f"Features shape: {features.shape}")
					# print(f"Features: {features}")

				features = features.cpu().numpy().astype(np.float32)

				asset_dict = {'features': features, 'coords': coords}
				save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
				mode = 'a'  # Append mode for subsequent saves

		except Exception as e:
			print(f"Error processing batch {count + 1}: {e}")
			raise  # Re-raise the exception to stop the loop if necessary
	
	# Print summary of saved samples
	if save_samples > 0:
		print(f"Saved {saved_samples}/{save_samples} random sample images for slide {slide_id}")
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1','prov_giga_path','virchow','virchow_v2','H-optimus-0','uni_v2','conch_v1_5'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--save_samples', type=int, default=10, help='Number of sample images to save per slide (0 to disable)')
args = parser.parse_args()


if __name__ == '__main__':
	
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	if args.save_samples > 0:
		os.makedirs(os.path.join(args.feat_dir, 'samples'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	  
	model = nn.DataParallel(model) # use all gpus avalable...
	model = model.to(device)

	total = len(bags_dataset)
	  
	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):

		try:
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			bag_name = slide_id+'.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
			print('\nprogress: {}/{}'.format(bag_candidate_idx, total))

			if not args.no_auto_skip and slide_id+'.pt' in dest_files:
				print('skipped {}'.format(slide_id))
				continue 

			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			time_start = time.time()

			if args.slide_ext == ".tiff":
				wsi = tiffslide.open_slide(slide_file_path) 
			else:
				wsi = openslide.open_slide(slide_file_path) 

			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi, 
										img_transforms=img_transforms)

			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
				
			try:
				output_file_path = compute_w_loader(output_path, loader=loader, model=model, device=device, model_name=args.model_name, slide_id=slide_id, feat_dir=args.feat_dir, save_samples=args.save_samples, verbose=1)
			except Exception as e:
				print("Error here")
				print(e)
					
			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

			with h5py.File(output_file_path, "r") as file:
				features = file['features'][:]
				print('features size: ', features.shape)
				print('coordinates size: ', file['coords'].shape)

			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))

		except Exception as e:
			print(f"An error occurred while saving the process list or accessing the index: {e}") 




