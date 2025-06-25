#!/bin/bash

# Function to run the CLAM pipeline (patching and embedding)
run_clam_pipeline() {
  local raw_slides_dir=$1
  local patch_save_dir=$2
  local csv_path=$3
  local slide_ext=$4
  local feat_dir=$5
  local microns_per_pixel=$6  # Accept microns_per_pixel as an additional argument

  echo "Running pipeline for slides in: $raw_slides_dir"

  # Build marking removal arguments
  marking_args=""
  if [ "${remove_markings:-false}" = "true" ]; then
    marking_args="--remove_markings"
    
    # Check for disabled color removals (only add if explicitly disabled)
    if [ "${disable_red_removal:-false}" = "true" ]; then
      marking_args="$marking_args --disable_red_removal"
    fi
    if [ "${disable_blue_removal:-false}" = "true" ]; then
      marking_args="$marking_args --disable_blue_removal"
    fi
    if [ "${disable_green_removal:-false}" = "true" ]; then
      marking_args="$marking_args --disable_green_removal"
    fi
    if [ "${disable_black_removal:-false}" = "true" ]; then
      marking_args="$marking_args --disable_black_removal"
    fi
    if [ "${disable_off_white_removal:-false}" = "true" ]; then
      marking_args="$marking_args --disable_off_white_removal"
    fi
  fi

  # Patching
  if [ "$microns_per_pixel" = "None" ]; then
    CUDA_VISIBLE_DEVICES=$cuda_devices_patch python CLAM/create_patches_fp.py \
      --source "$raw_slides_dir" \
      --save_dir "$patch_save_dir" \
      --microns_per_patch_edge "$microns_per_patch_edge" \
      --preset "$preset" \
      --seg \
      --patch \
      --stitch \
      --patch_level "$patch_level" \
      --max_patches "$max_patches" \
      $marking_args
  else
    CUDA_VISIBLE_DEVICES=$cuda_devices_patch python CLAM/create_patches_fp.py \
      --source "$raw_slides_dir" \
      --save_dir "$patch_save_dir" \
      --microns_per_patch_edge "$microns_per_patch_edge" \
      --preset "$preset" \
      --seg \
      --patch \
      --stitch \
      --patch_level "$patch_level" \
      --max_patches "$max_patches" \
      --microns_per_pixel "$microns_per_pixel" \
      $marking_args
  fi

  # 🚀 OPTIMIZED EMBEDDING - Using the new high-performance version
  echo "🚀 Starting optimized feature extraction..."
  
  # Build embedding arguments with optimization parameters
  embed_args="--data_h5_dir $patch_save_dir \
    --data_slide_dir $raw_slides_dir \
    --csv_path $csv_path \
    --slide_ext $slide_ext \
    --feat_dir $feat_dir \
    --model_name $model"
    
  # Add batch size if specified, otherwise let auto-detection handle it
  if [ -n "${batch_size}" ] && [ "${batch_size}" != "auto" ]; then
    embed_args="$embed_args --batch_size $batch_size"
  fi
  
  # Add write batch size for optimized I/O (default 10 if not specified)
  write_batch_size="${write_batch_size:-10}"
  embed_args="$embed_args --write_batch_size $write_batch_size"
  
  # Add sample saving (default 10 if not specified)
  save_samples="${save_samples:-10}"
  embed_args="$embed_args --save_samples $save_samples"
  
  # Run the optimized feature extraction
  CUDA_VISIBLE_DEVICES=$cuda_devices_patch python CLAM/extract_features_optimized.py $embed_args

  echo "✅ Completed optimized pipeline for slides in: $raw_slides_dir"
}

# Function to read YAML file
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

export OMP_NUM_THREADS=1

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file.yml>"
    echo ""
    echo "🚀 Now using OPTIMIZED feature extraction for 10-30x speedup!"
    echo "New optional config parameters:"
    echo "  batch_size: auto          # Auto-detect optimal batch size, or specify manually"
    echo "  write_batch_size: 10      # Batches to accumulate before writing (higher = faster I/O)"
    echo "  save_samples: 10          # Random sample images to save for visual inspection"
    echo ""
    exit 1
fi

# Read configuration
eval $(parse_yaml "$1")

# Check if microns_per_pixel is defined; set to None if not
microns_per_pixel="${microns_per_pixel:-None}"

echo "🚀 Starting OPTIMIZED CLAM pipeline with:"
echo "   Model: $model"
echo "   Batch size: ${batch_size:-auto-detected}"
echo "   Write batch size: ${write_batch_size:-10}"
echo "   Sample images: ${save_samples:-10}"

# Run pipeline for the dataset
run_clam_pipeline "$raw_slides_dir" "$patch_save_dir" "$csv_path" "$slide_ext" "$feat_dir" "$microns_per_pixel"
