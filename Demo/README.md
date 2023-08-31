


## Demo Quick Start
This repository reproduces our results on Terra Incognita, VLCS, and Urban Land, which is build upon Python3, Pytorch v1.12.1, and CUDA v10.2 on Ubuntu 18.04.
Please install all required packages by running:

```
pip install -r requirements.txt
```

## Results on Terra Incognita

Illustration of explanations from different data distributions in the Dog class of Terra Incognita dataset:

![demo](figures/expect_output.png)

We provide the pretrained weights of the ResNet-50 model trained by ERM (baseline) and DRE (ours), and demonstration code of the explanation results.
To reproduce the results on Terra Incognita, please the following notebook:
```
terra_reproduce.ipynb
```


## Acknowledgement

Part of our code is borrowed from the following repositories.

- [DomainBed](https://github.com/facebookresearch/DomainBed)
- [Captum](https://github.com/pytorch/captum)


We thank to the authors for releasing their codes. Please also consider citing their works.