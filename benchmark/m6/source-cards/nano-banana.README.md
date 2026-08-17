---
license: mit
task_categories:
- text-to-image
tags:
- synthetic-images
- nano-banana
- generated-images
pretty_name: Nano-Banana Generated Images
size_categories:
- 1K<n<10K
---

# Nano-Banana Generated Images

9,457 high-quality images generated using the Nano-Banana model (Google Gemini 2.5 Flash Image Preview).

## Dataset Overview

- **Total Images**: 9,457 images
- **Generation Method**: Nano-Banana (Google Gemini 2.5 Flash Image Preview)
- **Storage Format**: Optimized binary (Hugging Face Image type)
- **File Organization**: Normal large parquet files (not chunked)
- **License**: MIT

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier |
| `image` | Image | Binary image data (loads as PIL Image) |
| `format` | string | Image format (PNG, JPEG, etc.) |
| `mode` | string | Color mode (RGB, RGBA, etc.) |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `uploadtime` | string | Upload timestamp |

## Usage

```python
from datasets import load_dataset

# Load dataset - images are automatically decoded!
dataset = load_dataset("bitmind/nano-banana")

# Access images directly (no manual decoding needed!)
first_image = dataset['train'][0]['image']  # This is a PIL Image
first_image.show()

# Get image info
print(f"Format: {dataset['train'][0]['format']}")
print(f"Size: {dataset['train'][0]['width']}x{dataset['train'][0]['height']}")

# Iterate through dataset
for i, sample in enumerate(dataset['train']):
    if i < 5:  # Show first 5
        img = sample['image']  # Already a PIL Image
        print(f"Image {i+1}: {sample['format']} {sample['width']}x{sample['height']}")
```

## Benefits

- **Efficient Storage**: Binary format (no base64 overhead)
- **Fast Loading**: Direct PIL Image objects
- **Native Viewer Support**: Works with Hugging Face dataset viewer
- **Large Parquet Files**: Normal file organization (not micro-chunks)
