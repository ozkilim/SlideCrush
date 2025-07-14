#!/bin/bash

# Submit the Slurm job
echo "Submitting TracerX embedding job to Slurm..."
job_id=$(sbatch submit_embed.sbatch | grep -o '[0-9]*')

if [ $? -eq 0 ]; then
    echo "✅ Job submitted successfully!"
    echo "Job ID: $job_id"
    echo ""
    echo "Monitor your job with:"
    echo "  squeue -u $USER"
    echo "  squeue -j $job_id"
    echo ""
    echo "Check job output:"
    echo "  tail -f logs/embed_${job_id}.out"
    echo "  tail -f logs/embed_${job_id}.err"
    echo ""
    echo "Cancel job if needed:"
    echo "  scancel $job_id"
else
    echo "❌ Job submission failed!"
    exit 1
fi 