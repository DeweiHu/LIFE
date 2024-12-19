![](https://img.shields.io/badge/Language-python-brightgreen.svg)

# Unsupervised 3D Retinal Vessel Segmentation
### [2021 MICCAI] LIFE: A Generalizable Autodidactic Pipeline for 3D OCT-A Vessel Segmentation
---

### Method
In this work, we proposed to conduct the **self-fusion** algorithm on the neighboring depth images of 3D retinal OCT-A to create an auxiliary modality with cleaner vessels. Subsequently, we implement a dual-Unet architecture to map each en-face slide to its self-fusion counterpart. The latent image turns out to be a well-enhanced angiography that can be binarized with Otsu method.    
