# MicroNet: Improving Image Recognition with Extremely Low FLOPs (ICCV 2021)
A [pytorch](http://pytorch.org/) implementation of [MicroNet](https://arxiv.org/abs/2108.05894).
If you use this code in your research please consider citing
>@article{li2021micronet,
  title={MicroNet: Improving Image Recognition with Extremely Low FLOPs},
  author={Li, Yunsheng and Chen, Yinpeng and Dai, Xiyang and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Liu, Zicheng and Zhang, Lei and Vasconcelos, Nuno},
  journal={arXiv preprint arXiv:2108.05894},
  year={2021}
}
## Requirements

- Linux or macOS with Python ≥ 3.6.
- *Anaconda3*, *PyTorch ≥ 1.5* with matched [torchvision](https://github.com/pytorch/vision/)

## Models
Model | #Param | MAdds | Top-1 | download
--- |:---:|:---:|:---:|:---:
MicroNet-M3 | 2.6M | 21M  | 62.5 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m3.pth)
MicroNet-M2 | 2.4M | 12M  | 59.4 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m2.pth)
MicroNet-M1 | 1.8M | 6M  | 51.4 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m1.pth)
MicroNet-M0 | 1.0M | 4M  | 46.6 | [model](http://www.svcl.ucsd.edu/projects/micronet/assets/micronet-m0.pth)

## Evaluate MicroNet on ImageNet

Download the pretrained MicroNet M0-M3 with the link above. The scripts used for evaluation can be found [here](script). For example, if you want to test MicroNet-M3, you can use the following command.

```
sh scripts/eval_micronet_m3.sh /path/to/imagenet /path/to/output /path/to/pretrained_model
```

## Train MicroNet on ImageNet

The scripts used for training MicroNet M0-M3 can be found [here](script) and can be implemented as follows (You can choose to use different scripts for 2 gpu or 4 gpu training based on the resources you can access).
```
sh scripts/train_micronet_m3_4gpu.sh /path/to/imagenet /path/to/output
```
