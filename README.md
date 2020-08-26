# Metrics for Exposing the Biases of Content-Style Disentanglement
![overview](./assets/images/overview.png)

This repository contains the official implementation of the Distance Correlation (DC) and Information Over Bias (IOB) metrics proposed in [link]. The two metrics can be used to assess the level of disentanglement between **spatial** content and **vector** style representations. Both metrics are ready to use with PyTorch and TensorFlow implementations.

The repository is created by [Xiao Liu](https://github.com/xxxliu95)__\*__, [Spyridon Thermos](https://github.com/spthermo)__\*__, [Gabriele Valvano](https://github.com/gvalvano)__\*__, [Agisilaos Chartsias](https://github.com/agis85), [Alison O'Neil](https://www.eng.ed.ac.uk/about/people/dr-alison-oneil), and [Sotirios A. Tsaftaris](https://www.eng.ed.ac.uk/about/people/dr-sotirios-tsaftaris), as a result of the collaboration between [The University of Edinburgh](https://www.eng.ed.ac.uk/) and [Canon Medical Systems Europe](https://eu.medical.canon/).

# System Requirements
* Pytorch 1.5.1 or higher with GPU support
* Python 3.7.2 or higher
* TensorFlow r2.0 or higher with GPU support
* CUDA toolkit 10 or newer

**Note:** you need either PyTorch or TensorFlow to run the metrics, not both. 

# Metric 1: Distance Correlation (DC) - Independence

To compute the distance correlation between extracted content tensors (C) and style vectors (S), *i.e.* **DC(C,S)**, run the following command:

```python compute_DC.py --params```

To compute the distance correlation between extracted content tensors (C) and input image tensors (I), *i.e.* **DC(I,C)**, run the following command:

```python compute_DC.py --params```

To compute the distance correlation between extracted style vectors (S) and input image tensors (I), *i.e.* **DC(I,S)**, run the following command:

```python compute_DC.py --params```

# Metric 2: Information Over Bias (IOB) - Informativeness

To compute the information over bias between extracted content tensors (C) and input image tensors (I), *i.e.* **IOB(I,C)**, run the following command:

```python compute_IOB.py --params```

To compute the information over bias between extracted style vectors (S) and input image tensors (I), *i.e.* **IOB(I,S)**, run the following command:

```python compute_IOB.py --params```

# Applications
The two metrics have been tested on three popular models that exploit content-style disentanglement in the context of image-to-image translation, medical image segmentation, and pose estimation.

* MUNIT - [official github implementation](https://github.com/NVlabs/MUNIT), [paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Xun_Huang_Multimodal_Unsupervised_Image-to-image_ECCV_2018_paper.pdf)
* SDNet - [official github implementation](https://github.com/agis85/anatomy_modality_decomposition), [paper](https://arxiv.org/pdf/1903.09467.pdf)
* PNet  - [official github implementation](https://github.com/CompVis/unsupervised-disentangling), [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lorenz_Unsupervised_Part-Based_Disentangling_of_Object_Shape_and_Appearance_CVPR_2019_paper.pdf)

# Citation
If you find our metrics useful please cite the following paper:
```
@inproceedings{liu2020metrics,
  author       = "Xiao Liu and Spyridon Thermos and Gabriele Valvano and Agisilaos Chartsias and Alison O'Neil and Sotirios A. Tsaftaris",
  title        = "Metrics for Exposing the Biases of Content-Style Disentanglement",
  booktitle    = "arxiv",
  year         = "2020"
}
```

# License
All scripts are released under the MIT License.
