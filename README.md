# The Differentiable Lens: Compound Lens Search over Glass Surfaces and Materials for Object Detection

### [Paper](https://arxiv.org/abs/2212.04441) | [Project Page](https://light.princeton.edu/joint-lens-design)

#### Geoffroi Côté, Fahim Mannan, Simon Thibault, Jean-François Lalonde, Felix Heide

This is the official code repository for the paper: "The Differentiable Lens: Compound Lens Search over Glass Surfaces and Materials for Object Detection", presented at CVPR 2023.

![Lens Simulation Model](https://light.princeton.edu/wp-content/uploads/2023/04/joint-lens-design-simulation.jpg)

This repository provides code to
- model spherical lenses with arbitrary lens configurations based on their surface curvatures, spacings, and glass materials;
- evaluate their performance using a differentiable implementation of exact ray tracing;
- simulate physically accurate geometrical aberrations on input images that represent virtual scenes; and
- optimize them either in isolation or alongside downstream vision tasks.

In the CVPR 2023 work, this code was applied to object detection as a downstream computer vision task.

If you find our work useful in your research, please cite:

```
@inproceedings{cote2023differentiable,
  author = {C{\^o}t{\'e}, Geoffroi and Mannan, Fahim and Thibault, Simon and Lalonde, Jean-Fran{\c{c}}ois and Heide, Felix},
  title = {The Differentiable Lens: Compound Lens Search over Glass Surfaces and Materials for Object Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2023},
  pages = {},
}
```

## Requirements

The file ```environment.yml``` can be used to install a functional Conda environment. The environment was tested on Python 3.10 and Tensorflow 2.8, but any recent version of these packages should suffice.

```
conda env create -n joint-lens-design -f environment.yml
conda activate joint-lens-design
```

## Simulating aberrations

The sample script ```simulate_aberrations.py``` provides a simple demonstration of the proposed method. We provide four ```.yml``` files to model spherical lenses with 1, 2, 3, and 4 refractive elements. The latter three correspond to the baseline lenses in the paper. For a complete list of command-line arguments, try:

```
python simulate_aberrations --help
```

For any question or advice, please reach out to me at gcote[at]princeton[dot]edu.
