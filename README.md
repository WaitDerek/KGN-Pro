# Keypoint-GraspNet (KGN)


## Installation

Please follow [INSTALL.md](docs/INSTALL.md) to prepare for the environment.



## Train and evaluate

### Data generation

The dataset used in the papers can be downloaded from the links: [sinlge-object](https://www.dropbox.com/s/gfcddf7awkjw1wy/ps_grasp_single_1k.zip?dl=0) and [multi-object](https://www.dropbox.com/s/kmysg23usmaycmf/ps_grasp_multi_1k.zip?dl=0). Download, extract, and put them in the ``./data/`` folder.



Alternatively, you can also generate the data by yourself. For single-object data generation:

```bash
python main_data_generate.py --config_file lib/data_generation/ps_grasp_single.yaml
```

Multi-object data generation:

```bash
python main_data_generate.py --config_file lib/data_generation/ps_grasp_multi.yaml
```



### Train

First download pretrained [ctdet_coco_dla_2x](https://github.com/xingyizhou/CenterNet) model following the instruction. Put it under ``./models/``  folder.

Then run the training code.

```bash
bash experiments/train_kgnv{1|2}.sh {single|multi}
```

``single/multi``: Train on single- or multi-object data. 




### Evaluation

```bash
bash experiments/test_kgnv{1|2}.sh {single|multi} {single|multi}
```

First ``single/multi``: Evaluate the weight trained on single- or multi-object data. 

Second ``single/multi``: Evaluate on single- or multi-object data. 


### Acknowledgment
Thanks for these awesome works [KGN](https://github.com/ivalab/KGN) and [EPro-pnp-v2](https://github.com/tjiiv-cprg/EPro-PnP-v2)
