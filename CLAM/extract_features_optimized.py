import numpy as np
import time
import os
import argparse
import pdb
from functools import partial
import random
from collections import defaultdict

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
import tiffslide
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PerformanceProfiler:
	"""Comprehensive performance profiler for identifying bottlenecks"""
	
	def __init__(self):
		self.timings = defaultdict(list)
		self.counters = defaultdict(int)
		self.memory_usage = []
		self.gpu_memory_usage = []
		self.start_times = {}
		
	def reset(self):
		"""Reset profiler for next slide"""
		self.timings = defaultdict(list)
		self.counters = defaultdict(int)
		self.memory_usage = []
		self.gpu_memory_usage = []
		self.start_times = {}
		
	def start_timer(self, name):
		"""Start timing an operation"""
		self.start_times[name] = time.perf_counter()
		
	def end_timer(self, name):
		"""End timing an operation"""
		if name in self.start_times:
			duration = time.perf_counter() - self.start_times[name]
			self.timings[name].append(duration)
			del self.start_times[name]
			return duration
		return 0
	
	def count_operation(self, name, count=1):
		"""Count occurrences of an operation"""
		self.counters[name] += count
		
	def record_memory_usage(self):
		"""Record current memory usage"""
		try:
			import psutil
			process = psutil.Process()
			cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
			self.memory_usage.append(cpu_mem)
		except ImportError:
			# psutil not available, skip CPU memory tracking
			pass
		
		if torch.cuda.is_available():
			gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
			self.gpu_memory_usage.append(gpu_mem)
		
	def get_summary(self, slide_id=None):
		"""Generate comprehensive performance summary"""
		summary = []
		summary.append("=" * 80)
		title = f"🔍 PERFORMANCE PROFILING SUMMARY"
		if slide_id:
			title += f" - {slide_id}"
		summary.append(title)
		summary.append("=" * 80)
		
		# Timing analysis
		total_times = {}
		for name, times in self.timings.items():
			total_time = sum(times)
			avg_time = total_time / len(times)
			total_times[name] = total_time
			summary.append(f"⏱️  {name}:")
			summary.append(f"   Total: {total_time:.3f}s | Avg: {avg_time:.4f}s | Calls: {len(times)}")
			if len(times) > 1:
				summary.append(f"   Min: {min(times):.4f}s | Max: {max(times):.4f}s")
		
		# Find bottlenecks (operations taking >5% of total time)
		if total_times:
			total_runtime = sum(total_times.values())
			summary.append(f"\n🚨 BOTTLENECK ANALYSIS (Total Runtime: {total_runtime:.3f}s):")
			bottlenecks = [(name, time_val) for name, time_val in total_times.items() 
						  if time_val / total_runtime > 0.05]
			bottlenecks.sort(key=lambda x: x[1], reverse=True)
			
			for name, time_val in bottlenecks:
				percentage = (time_val / total_runtime) * 100
				summary.append(f"   🔥 {name}: {time_val:.3f}s ({percentage:.1f}%)")
		
		# Operation counts
		if self.counters:
			summary.append(f"\n📊 OPERATION COUNTS:")
			for name, count in sorted(self.counters.items()):
				summary.append(f"   {name}: {count:,}")
		
		# Memory analysis
		if self.memory_usage:
			max_cpu = max(self.memory_usage)
			avg_cpu = sum(self.memory_usage) / len(self.memory_usage)
			summary.append(f"\n💾 MEMORY USAGE:")
			summary.append(f"   CPU Memory - Max: {max_cpu:.1f}MB | Avg: {avg_cpu:.1f}MB")
			
		if self.gpu_memory_usage:
			max_gpu = max(self.gpu_memory_usage)
			avg_gpu = sum(self.gpu_memory_usage) / len(self.gpu_memory_usage)
			summary.append(f"   GPU Memory - Max: {max_gpu:.1f}MB | Avg: {avg_gpu:.1f}MB")
		
		# Performance recommendations
		summary.append(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
		if 'data_loading' in total_times and total_times['data_loading'] > total_runtime * 0.2:
			summary.append("   • Data loading is slow - consider more workers or faster storage")
		if 'model_inference' in total_times and total_times['model_inference'] > total_runtime * 0.6:
			summary.append("   • Model inference dominates - consider larger batch sizes or model optimization")
		if 'hdf5_writing' in total_times and total_times['hdf5_writing'] > total_runtime * 0.1:
			summary.append("   • HDF5 writing is slow - consider larger write batches or SSD storage")
		if 'wsi_reading' in total_times and total_times['wsi_reading'] > total_runtime * 0.3:
			summary.append("   • WSI reading is slow - consider patch pre-extraction or faster storage")
		
		summary.append("=" * 80)
		return "\n".join(summary)

# Global profiler instance
profiler = PerformanceProfiler()

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

def compute_w_loader_optimized(output_path, loader, model, device, model_name, slide_id, feat_dir, save_samples=10, verbose=1, write_batch_size=10, enable_profiling=True):
	"""
	Optimized feature extraction with batched writes and mixed precision.
	
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
		write_batch_size: number of batches to accumulate before writing to disk
	"""
	if enable_profiling:
		profiler.start_timer('total_computation')
		profiler.record_memory_usage()
	
	def profile_if_enabled(func_name, *args, **kwargs):
		"""Helper to conditionally call profiler functions"""
		if enable_profiling:
			return getattr(profiler, func_name)(*args, **kwargs)
		return None
	
	if verbose > 0:
		status_msg = "with profiling enabled" if enable_profiling else "without profiling"
		print(f'🔍 Processing a total of {len(loader)} batches {status_msg}')

	model.eval()  # Ensure the model is in evaluation mode
	
	# Setup for random sampling across batches
	saved_samples = 0
	total_batches = len(loader)
	if save_samples > 0 and total_batches > 0:
		sample_probability = min(1.0, save_samples / total_batches)
		samples_dir = os.path.join(feat_dir, 'samples', slide_id)
		os.makedirs(samples_dir, exist_ok=True)
	
	# Initialize storage for batched writes
	all_features = []
	all_coords = []
	write_mode = 'w'
	
	# Enable mixed precision
	scaler = torch.cuda.amp.GradScaler()
	
	for count, data in enumerate(tqdm(loader, desc=f"Processing {slide_id}")):
		try:
			profile_if_enabled('start_timer', 'batch_processing')
			profile_if_enabled('start_timer', 'data_loading')
			
			# Mixed precision inference with no_grad for memory efficiency
			with torch.no_grad(), autocast():
				batch = data['img']
				coords = data['coord'].numpy().astype(np.int32)
				profile_if_enabled('end_timer', 'data_loading')
				profile_if_enabled('count_operation', 'patches_processed', batch.shape[0])
				
				# Randomly sample images from across all batches
				if save_samples > 0 and saved_samples < save_samples:
					if random.random() < sample_probability or saved_samples == 0:
						profile_if_enabled('start_timer', 'sample_saving')
						batch_size = batch.shape[0]
						random_idx = random.randint(0, batch_size - 1)
						
						sample_img = batch[random_idx:random_idx+1]
						sample_coord = coords[random_idx:random_idx+1]
						
						save_random_sample(sample_img, sample_coord, slide_id, feat_dir, model_name, saved_samples)
						saved_samples += 1
						profile_if_enabled('end_timer', 'sample_saving')

				# Move to device with non-blocking transfer
				profile_if_enabled('start_timer', 'data_transfer')
				batch = batch.to(device, non_blocking=True)
				profile_if_enabled('end_timer', 'data_transfer')
				
				# Forward pass
				profile_if_enabled('start_timer', 'model_inference')
				features = model(batch)
				profile_if_enabled('end_timer', 'model_inference')
				
				# Model-specific feature processing
				profile_if_enabled('start_timer', 'feature_processing')
				if model_name == "virchow":
					class_token = features[:, 0]    # size: n x 1280
					patch_tokens = features[:, 1:]  # size: n x 256 x 1280
					features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: n x 2560

				elif model_name == "virchow_v2":
					class_token = features[:, 0]    # size: 1 x 1280
					patch_tokens = features[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens
					features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560

				# Convert to numpy and accumulate
				features = features.cpu().numpy().astype(np.float32)
				profile_if_enabled('end_timer', 'feature_processing')
				
				profile_if_enabled('start_timer', 'feature_accumulation')
				all_features.append(features)
				all_coords.append(coords)
				profile_if_enabled('end_timer', 'feature_accumulation')
				
				# Write in batches to reduce I/O overhead
				if len(all_features) >= write_batch_size or count == len(loader) - 1:
					profile_if_enabled('start_timer', 'hdf5_writing')
					profile_if_enabled('count_operation', 'hdf5_write_operations')
					
					# Combine accumulated batches
					profile_if_enabled('start_timer', 'feature_combining')
					combined_features = np.vstack(all_features)
					combined_coords = np.vstack(all_coords)
					profile_if_enabled('end_timer', 'feature_combining')
					
					# Write to HDF5
					asset_dict = {'features': combined_features, 'coords': combined_coords}
					save_hdf5(output_path, asset_dict, attr_dict=None, mode=write_mode)
					write_mode = 'a'  # Switch to append mode after first write
					profile_if_enabled('end_timer', 'hdf5_writing')
					
					# Clear accumulated data and force garbage collection
					profile_if_enabled('start_timer', 'memory_cleanup')
					all_features.clear()
					all_coords.clear()
					del combined_features, combined_coords
					gc.collect()
					profile_if_enabled('end_timer', 'memory_cleanup')
					profile_if_enabled('record_memory_usage')
					
					if verbose > 1:
						print(f"Written batch {count // write_batch_size + 1}")

			profile_if_enabled('end_timer', 'batch_processing')
			
		except Exception as e:
			print(f"Error processing batch {count + 1}: {e}")
			# Clean up before raising
			all_features.clear()
			all_coords.clear()
			gc.collect()
			raise

	# Print summary of saved samples
	if save_samples > 0:
		print(f"Saved {saved_samples}/{save_samples} random sample images for slide {slide_id}")
	
	profile_if_enabled('end_timer', 'total_computation')
	return output_path

def get_optimal_batch_size(model_name, target_patch_size=224):
	"""
	Get optimal batch size based on model and available GPU memory.
	"""
	# Conservative batch sizes for different models
	batch_sizes = {
		'resnet50_trunc': 512,
		'uni_v1': 256,
		'uni_v2': 64,  # Large model, needs smaller batches
		'conch_v1': 256,
		'prov_giga_path': 128,
		'virchow': 64,
		'virchow_v2': 32,  # Very large model
		'H-optimus-0': 128
	}
	
	base_batch_size = batch_sizes.get(model_name, 256)
	
	# Adjust for patch size
	if target_patch_size > 224:
		base_batch_size = base_batch_size // 2
	
	# Check GPU memory and adjust
	if torch.cuda.is_available():
		gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
		if gpu_memory_gb < 12:  # Less than 12GB
			base_batch_size = base_batch_size // 2
		elif gpu_memory_gb > 24:  # More than 24GB
			base_batch_size = min(base_batch_size * 2, 1024)
	
	return base_batch_size

def get_optimal_dataloader_kwargs(device):
	"""
	Get optimal DataLoader parameters based on system capabilities.
	"""
	if device.type == "cuda":
		# Optimal settings for GPU
		kwargs = {
			'num_workers': 4,  # Conservative for stability
			'pin_memory': True,
			'prefetch_factor': 2,
			'persistent_workers': True,
			'drop_last': False
		}
	else:
		# CPU settings
		kwargs = {
			'num_workers': 2,
			'pin_memory': False,
			'prefetch_factor': 2
		}
	
	return kwargs

parser = argparse.ArgumentParser(description='Optimized Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', 
                   choices=['resnet50_trunc', 'uni_v1', 'conch_v1','prov_giga_path','virchow','virchow_v2','H-optimus-0','uni_v2'])
parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-detected if not specified)')
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--save_samples', type=int, default=10, help='Number of sample images to save per slide (0 to disable)')
parser.add_argument('--write_batch_size', type=int, default=10, help='Number of batches to accumulate before writing to disk')
parser.add_argument('--mixed_precision', default=True, action='store_true', help='Use mixed precision (default: True)')
parser.add_argument('--profile', default=True, action='store_true', help='Enable detailed performance profiling (default: True)')
parser.add_argument('--no_profile', dest='profile', action='store_false', help='Disable performance profiling')

args = parser.parse_args()

if __name__ == '__main__':
	
	print('🚀 Initializing optimized feature extraction...')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError("CSV path is required")

	bags_dataset = Dataset_All_Bags(csv_path)
	
	# Create directories
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	if args.save_samples > 0:
		os.makedirs(os.path.join(args.feat_dir, 'samples'), exist_ok=True)
	
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	# Load model
	print(f'📋 Loading model: {args.model_name}')
	if args.profile:
		profiler.start_timer('model_loading')
	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
	model = model.eval()
	if args.profile:
		profiler.end_timer('model_loading')
	
	# Optimize model for inference
	if hasattr(torch, 'jit') and device.type == "cuda":
		try:
			# Try to optimize with TorchScript (may not work for all models)
			model = torch.jit.optimize_for_inference(model)
			print("✅ Model optimized with TorchScript")
		except:
			print("⚠️  TorchScript optimization not available for this model")
	
	# Use DataParallel if multiple GPUs available
	if torch.cuda.device_count() > 1:
		print(f"🔥 Using {torch.cuda.device_count()} GPUs with DataParallel")
		model = nn.DataParallel(model)
	
	model = model.to(device)
	
	# Auto-detect optimal batch size if not specified
	if args.batch_size is None:
		args.batch_size = get_optimal_batch_size(args.model_name, args.target_patch_size)
		print(f"🎯 Auto-detected optimal batch size: {args.batch_size}")
	else:
		print(f"📊 Using specified batch size: {args.batch_size}")

	total = len(bags_dataset)
	print(f"📁 Processing {total} slides")
	
	# Get optimal dataloader settings
	loader_kwargs = get_optimal_dataloader_kwargs(device)
	print(f"⚙️  DataLoader settings: {loader_kwargs}")

	# Process each slide
	start_time = time.time()
	processed_slides = 0
	
	for bag_candidate_idx in tqdm(range(total), desc="Overall Progress"):
		try:
			slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
			bag_name = slide_id + '.h5'
			h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
			slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
			
			print(f'\n🔄 Processing slide {bag_candidate_idx + 1}/{total}: {slide_id}')

			# Skip if already processed
			if not args.no_auto_skip and slide_id + '.pt' in dest_files:
				print(f'⏭️  Skipped {slide_id} (already processed)')
				continue 

			output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
			slide_start_time = time.time()

			# Load WSI
			if args.profile:
				profiler.start_timer('wsi_loading')
			if args.slide_ext == ".tiff":
				wsi = tiffslide.open_slide(slide_file_path) 
			else:
				wsi = openslide.open_slide(slide_file_path) 
			if args.profile:
				profiler.end_timer('wsi_loading')

			# Create dataset and loader
			if args.profile:
				profiler.start_timer('dataset_creation')
			dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
										wsi=wsi, 
										img_transforms=img_transforms)

			loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
			if args.profile:
				profiler.end_timer('dataset_creation')
			
			print(f"📦 Dataset size: {len(dataset)} patches, {len(loader)} batches")
				
			# Extract features
			try:
				if args.profile:
					profiler.start_timer('slide_processing')
				output_file_path = compute_w_loader_optimized(
					output_path, 
					loader=loader, 
					model=model, 
					device=device, 
					model_name=args.model_name, 
					slide_id=slide_id, 
					feat_dir=args.feat_dir, 
					save_samples=args.save_samples,
					verbose=1,
					write_batch_size=args.write_batch_size,
					enable_profiling=args.profile
				)
				if args.profile:
					profiler.end_timer('slide_processing')
			except Exception as e:
				print(f"❌ Error processing slide {slide_id}: {e}")
				continue
					
			slide_elapsed = time.time() - slide_start_time
			print(f'✅ Features extracted in {slide_elapsed:.1f}s')
			
			# Show profiling results for this slide
			if args.profile:
				print(f"\n{profiler.get_summary(slide_id)}")
				profiler.reset()  # Reset for next slide

			# Verify and save features as .pt file
			try:
				with h5py.File(output_file_path, "r") as file:
					features = file['features'][:]
					coords = file['coords'][:]
					print(f'📊 Features shape: {features.shape}, Coords shape: {coords.shape}')

				features_tensor = torch.from_numpy(features)
				bag_base, _ = os.path.splitext(bag_name)
				pt_path = os.path.join(args.feat_dir, 'pt_files', bag_base + '.pt')
				torch.save(features_tensor, pt_path)
				print(f'💾 Saved features to {pt_path}')
				
				processed_slides += 1
				
			except Exception as e:
				print(f"❌ Error saving features for {slide_id}: {e}")
				continue
			
			# Clean up WSI object
			del wsi
			gc.collect()

		except Exception as e:
			print(f"❌ Error processing slide {bag_candidate_idx}: {e}") 
			continue

	# Final summary
	total_time = time.time() - start_time
	avg_time = total_time / max(processed_slides, 1)
	
	print(f"\n🎉 Feature extraction completed!")
	print(f"📈 Processed: {processed_slides}/{total} slides")
	print(f"⏱️  Total time: {total_time:.1f}s")
	print(f"⚡ Average time per slide: {avg_time:.1f}s")
	
	if processed_slides > 0:
		print(f"🚀 Throughput: {processed_slides / (total_time / 3600):.1f} slides/hour") 