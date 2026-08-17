---
license: cc-by-4.0
dataset_info:
  features:
  - name: image
    dtype: image
  - name: generator
    dtype: string
  - name: uid
    dtype: string
  - name: labels
    list:
    - name: label
      dtype: string
    - name: points
      list:
        list: float64
  - name: original_prompt
    dtype: string
  - name: positive_prompt
    dtype: string
  - name: negative_prompt
    dtype: string
  - name: guidance_scale
    dtype: float64
  - name: num_inference_steps
    dtype: int64
  - name: scheduler
    dtype: string
  - name: seed
    dtype: int64
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: image_format
    dtype: string
  - name: jpeg_quality
    dtype: int64
  - name: chroma_subsampling
    dtype: string
  splits:
  - name: labeled_train
    num_bytes: 1229331054
    num_examples: 918
  - name: labeled_test
    num_bytes: 3490081723
    num_examples: 2419
  - name: unlabeled_train
    num_bytes: 34599400559
    num_examples: 24013
  - name: unlabeled_test
    num_bytes: 35214906257
    num_examples: 24638
configs:
- config_name: default
  data_files:
  - split: labeled_train
    path: data/labeled_train-*
  - split: labeled_test
    path: data/labeled_test-*
  - split: unlabeled_train
    path: data/unlabeled_train-*
  - split: unlabeled_test
    path: data/unlabeled_test-*
pretty_name: X-AIGD
---

# X-AIGD

<p align="center">
  <a href="https://arxiv.org/abs/2601.19430"><img src="https://img.shields.io/badge/arXiv-2601.19430-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/Coxy7/X-AIGD"><img src="https://img.shields.io/badge/GitHub-X--AIGD-blue?logo=github" alt="GitHub"></a>
</p>

X-AIGD is a fine-grained benchmark designed for **eXplainable AI-Generated image Detection**. It provides pixel-level human annotations of perceptual artifacts in AI-generated images, spanning low-level distortions, high-level semantics, and cognitive-level counterfactuals, aiming to advance robust and explainable AI-generated image detection methods.

For more details, please refer to our paper: [Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection](https://arxiv.org/abs/2601.19430).

## 🔄 Updates

- `2026.03.16`: Uploaded the real-image metadata and code for reconstructing the real-image set.
- `2026.03.12`: We noticed that some images in the `labeled_test` split with generator `SD_3.5L-raw` were mismatched with the labels, and replaced them with the correct images.
- `2026.02.11`: Uploaded unlabeled data splits.
- `2026.02.09`: Uploaded labeled data splits.

## 🎨 Artifact Taxonomy

We define a comprehensive artifact taxonomy comprising 3 levels and 7 specific categories to capture the diverse range of perceptual artifacts in AI-generated images.

<p align="center">
  <img src="taxonomy.jpg" width="800">
</p>

*   **Low-level Distortions:** `low-level-edge_shape`, `low-level-texture`, `low-level-color`, `low-level-symbol`.
*   **High-level Semantics:** `high-level-semantics`.
*   **Cognitive-level Counterfactuals:** `cognitive-level-commonsense`, `cognitive-level-physics`.

## 🚀 Dataset Contents

This repository hosts the **pixel-level annotated subset** of X-AIGD, which includes over 18,000 artifact instances across 3,000+ labeled samples, along with an **unlabeled** subset.

### Data Splits
- `labeled_train`, `labeled_test`: the annotated subsets of AI-generated images with pixel-level artifact labels (total 3,337 samples).
- `unlabeled_train`, `unlabeled_test`: the unannotated subsets of AI-generated images without artifact labels (total 48,651 samples).
- Real images: sourced from `SA-1B`, `MSCOCO`, `LAION-2B-en-aesthetic`, and `CC3M` (total 4,000 samples).

### Data Fields (for the AI-generated images)

- `image`: The AI-generated image (raw images with **PNG** format).
- `generator`: Name of the text-to-image generator.
- `uid`: Unique identifier for the image.
- `labels`: List of human-annotated artifacts, each containing:
    - `label`: Category of the artifact (e.g., `low-level-edge_shape`, `high-level-semantics`).
    - `points`: Polygon coordinates `[[x1, y1], [x2, y2], ...]` localizing the artifact.
- `original_prompt`, `positive_prompt`, `negative_prompt`: Text prompts used for generation.
- `num_inference_steps`, `guidance_scale`, `seed`, `scheduler`: Generation parameters.
- `width`, `height`: Image resolution.
- `image_format`, `jpeg_quality`, `chroma_subsampling`: Image compression details of the _corresponding real image_ (used for optional compression alignment).

### UID Correspondence & Train-Test Splitting

- Each AI-generated (fake) image is generated based on the caption of a real image and inherits its `uid` from the corresponding real image.
  - Fake images across different generators with the same `uid` share the same original prompt.
  - Real and fake images with the same `uid` are expected to be semantically similar.
- The train-test splitting is based on `uid`, as defined in `uid_splits.csv` in this repository.

### Real Images

Due to third-party license restrictions, the real-image set is not hosted directly in this repository. Instead, we provide:
- `real_image_metadata.csv`: metadata for all real images
- `download_real_images.py`: downloader for reconstructing the real-image set


Example running command for the downloader (please run `python download_real_images.py -h` for more details):

```bash
# pip install datasets huggingface_hub pillow
python download_real_images.py \
  --metadata real_image_metadata.csv \
  --output-dir ./real_images \
  --source all \
  --sa1b-index-file /path/to/sa-1b.txt
```

**Important notes**:
- **SA-1B**: To obtain the SA-1B index file (`sa-1b.txt` in the example), visit the [official website](https://ai.meta.com/datasets/segment-anything-downloads/).
- **Reproducibility**: We notice that some images from LAION and CC3M are no longer accessible via their original URLs, or they have been modified (e.g., compressed) since we collected them. To reconstruct the exact full real-image set used in our paper, researchers may request access to the [X-AIGD-real](https://huggingface.co/datasets/Coxy7/X-AIGD-real) repository. Access is granted strictly for _non-commercial research purposes_ only.


## 📖 Usage Example

```python
from datasets import load_dataset

# Load the labeled test split (AI-generated images with artifact annotations)
ds = load_dataset("Coxy7/X-AIGD", split="labeled_test")

# Access an example
sample = ds[0]
print(f"Generator: {sample['generator']}")
print(f"UID: {sample['uid']}")

# Access artifact labels and polygon localization
for artifact in sample["labels"]:
    print(f"Artifact category: {artifact['label']}")
    print(f"Polygon points: {artifact['points']}")

# The image is a PIL object
# sample["image"].save("img.png")
```

## 📝 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{xiao2026unveiling,
  title={Unveiling Perceptual Artifacts: A Fine-Grained Benchmark for Interpretable AI-Generated Image Detection},
  author={Xiao, Yao and Chen, Weiyan and Chen, Jiahao and Cao, Zijie and Deng, Weijian and Yang, Binbin and Dong, Ziyi and Ji, Xiangyang and Ke, Wei and Wei, Pengxu and Lin, Liang},
  journal={arXiv preprint arXiv:2601.19430},
  year={2026}
}
```

## 📄 License

The dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
