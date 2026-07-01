#!/bin/bash
#SBATCH --job-name=TracerX_embed
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --chdir=/nemo/project/proj-tracerx-lung/tctProjects/kilimo/SlideCrush
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "============================================"

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo "============================================"

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the correct working directory
cd /nemo/project/proj-tracerx-lung/tctProjects/kilimo/SlideCrush
echo "Working directory: $(pwd)"

# Run the embed script with the TracerX configuration
echo "Starting embed.sh with TracerX configuration..."
./embed.sh /nemo/project/proj-tracerx-lung/tctProjects/kilimo/analysis/SlideCrush/configs/TracerX/CPTAC.yaml

echo "============================================"
echo "Job completed at: $(date)"
echo "============================================" 