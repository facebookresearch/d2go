# DETR and Deformable DETR in D2Go

This project extend D2Go with [DETR](https://github.com/facebookresearch/detr) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) models. The pretrained models with efficient backbone are provided.

## Usage

Please install D2Go following the [instructions](../README.md). Then install this extension:

```bash
cd projects/detr/
python setup.py install
```

### Evaluating Pretrained Models

Please use the `tools/train_net.py` in the main directory as the entry point. The pretrained model can be evaluated using

```bash
python train_net.py --runner detr.runner.DETRRunner --eval-only --config configs/deformable_detr_fbnetv3a_bs16.yaml  MODEL.WEIGHTS https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/252811934/model_final.pth
```

### Training

Please use the `tools/train_net.py` in the main directory as the entry point and pass the runner as `detr.runner.DETRRunner`.

```bash
python train_net.py --runner detr.runner.DETRRunner --config configs/deformable_detr_fbnetv3a_bs16.yaml 
```

### Pretrained Models

| name                                                         | box AP | model id  | download                                                     |
| ------------------------------------------------------------ | ------ | --------- | ------------------------------------------------------------ |
| [Deformable-DETR-FBNetV3A](./configs/deformable_detr_fbnetv3a_bs16.yaml) | 27.53  | 252811934 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/252811934/model_final.pth)\|[mertrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/252811934/metrics.json) |

