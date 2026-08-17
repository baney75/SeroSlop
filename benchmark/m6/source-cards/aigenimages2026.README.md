---
license: cc-by-4.0
---

# Dataset Card for AIGenImages2026


## Dataset Sources

- **Repository:** [WildFC GitHub Repository](https://github.com/mever-team/WildFC)
- **Project Page:** [WildFC Project Page](https://mever-team.github.io/WildFC/)
- **Paper:** [Automated In-the-Wild Data Collection for Continual AI Generated Image Detection (arXiv)](https://arxiv.org/pdf/2605.02567)


## Dataset Description

AIGenImages2026 is a continually evolving benchmark dataset of AI-generated images created from recent text-to-image generative models released throughout 2025. The dataset was designed to support research in AI-generated image detection (AID), continual learning, robustness evaluation, and distribution shift analysis.

The dataset contains 5,439 AI-generated images produced by 19 contemporary generative models. Images were generated using diverse prompt strategies emphasizing realism, compositional reasoning, stylistic variation, and real-world semantics.

AIGenImages2026 was introduced as part of the continual adaptation framework proposed in the paper *Automated In-the-Wild Data Collection for Continual AI Generated Image Detection*. The dataset is intended to evaluate detector robustness against rapidly evolving generative models and emerging synthesis artifacts.

The dataset includes chronological generator metadata to facilitate temporal benchmarking and continual learning research.


## Dataset Structure

The dataset contains 5,439 AI-generated images generated from 19 recent text-to-image models.

### Dataset Splits

- **Training set:** 4,880 images
- **Test set:** 559 images


### Included Metadata

Each sample:
- Image file
- Generator/model name
- Prompt
- Split assignment (train/test)

---

## Citation

If you use AIGenImages2026 in your research, please cite the following paper:

### BibTeX

```bibtex
@inproceedings{pantsios2026automated,
  title={Automated In-the-Wild Data Collection for Continual AI Generated Image Detection},
  author={Pantsios, Athanasios and Karageorgiou, Dimitrios and Koutlis, Christos and Karantaidis, George and Papadopoulou, Olga and Papadopoulos, Symeon},
  booktitle={The 5th ACM International Workshop on Multimedia AI against Disinformation (MAD '26)},
  year={2026},
  doi={10.1145/3810988.3812662}
}