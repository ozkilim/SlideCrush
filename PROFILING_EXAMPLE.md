# 🔍 Performance Profiling Guide

## How to Use Profiling

The optimized feature extraction script now includes comprehensive performance profiling to identify bottlenecks and optimization opportunities.

### Basic Usage

```bash
# Enable profiling (default)
python CLAM/extract_features_optimized.py \
  --data_h5_dir /path/to/patches \
  --data_slide_dir /path/to/slides \
  --feat_dir /path/to/features \
  --model_name uni_v2 \
  --csv_path /path/to/slides.csv \
  --profile

# Disable profiling for maximum speed
python CLAM/extract_features_optimized.py \
  --data_h5_dir /path/to/patches \
  --data_slide_dir /path/to/slides \
  --feat_dir /path/to/features \
  --model_name uni_v2 \
  --csv_path /path/to/slides.csv \
  --no_profile
```

### Or via embed.sh

Simply add this to your YAML config:
```yaml
model: uni_v2
batch_size: auto
write_batch_size: 10
save_samples: 10
# Profiling is enabled by default
```

## Example Profiling Output

After processing, you'll see a detailed performance summary:

```
================================================================================
🔍 PERFORMANCE PROFILING SUMMARY
================================================================================
⏱️  model_loading:
   Total: 2.451s | Avg: 2.4510s | Calls: 1

⏱️  wsi_loading:
   Total: 0.123s | Avg: 0.0123s | Calls: 10

⏱️  dataset_creation:
   Total: 0.089s | Avg: 0.0089s | Calls: 10

⏱️  data_loading:
   Total: 15.234s | Avg: 0.0152s | Calls: 1000

⏱️  model_inference:
   Total: 125.678s | Avg: 0.1257s | Calls: 1000

⏱️  feature_processing:
   Total: 2.345s | Avg: 0.0023s | Calls: 1000

⏱️  hdf5_writing:
   Total: 8.901s | Avg: 0.8901s | Calls: 10

🚨 BOTTLENECK ANALYSIS (Total Runtime: 154.821s):
   🔥 model_inference: 125.678s (81.2%)
   🔥 data_loading: 15.234s (9.8%)
   🔥 hdf5_writing: 8.901s (5.7%)

📊 OPERATION COUNTS:
   hdf5_write_operations: 10
   patches_processed: 25,600

💾 MEMORY USAGE:
   CPU Memory - Max: 8,945.2MB | Avg: 6,123.4MB
   GPU Memory - Max: 11,234.7MB | Avg: 9,876.5MB

💡 OPTIMIZATION RECOMMENDATIONS:
   • Model inference dominates - consider larger batch sizes or model optimization
   • Data loading is slow - consider more workers or faster storage
================================================================================
```

## Interpreting Results

### Key Metrics to Watch:

1. **🔥 Bottleneck Analysis**: Shows which operations consume the most time
   - **Model Inference > 60%**: Normal for GPU inference
   - **Data Loading > 20%**: Consider faster storage or more workers
   - **HDF5 Writing > 10%**: Increase `write_batch_size`
   - **WSI Reading > 30%**: Consider patch pre-extraction

2. **Memory Usage**: Monitor for OOM issues
   - **High GPU Memory**: Reduce batch size
   - **High CPU Memory**: Reduce num_workers or write_batch_size

### Common Optimization Strategies:

| Issue | Solution |
|-------|----------|
| High data_loading time | Increase num_workers, use SSD storage |
| High hdf5_writing time | Increase write_batch_size (10→50) |
| High model_inference time | Increase batch_size, use larger GPU |
| High wsi_loading time | Use faster storage, consider patch pre-extraction |
| High memory usage | Reduce batch_size or num_workers |

## Performance Tuning Examples

### For Fast SSDs + High Memory:
```yaml
batch_size: 128          # Larger batches
write_batch_size: 50     # Fewer writes
save_samples: 5          # Fewer samples
```

### For Slow Storage:
```yaml
batch_size: 64           # Smaller batches
write_batch_size: 5      # More frequent writes
save_samples: 0          # Disable samples
```

### For Large GPU Memory (>24GB):
```yaml
batch_size: 256          # Much larger batches
write_batch_size: 20     # Bigger write chunks
save_samples: 15         # More samples if desired
```

## Share Results for Further Optimization

When sharing profiling results, include:
1. The full profiling summary output
2. Your hardware specs (GPU, CPU, storage type)
3. Model being used
4. Typical slide sizes and patch counts

This helps identify additional optimization opportunities specific to your setup! 