# D2Go Model Zoo and Baselines

## Introduction

This page holds a reference for example configs, pretrained models and training/evaluation metrics. You can access these models from code using d2go.model_zoo API.

### How to

- Train: the "name" column contains a link to the config file. Running `tools/train_net.py --num-gpus 8` with the config file will reproduce the corresponding model.
- Evaluation: Running  `tools/train_net.py --eval-only` with the config file and pretrained model will evaluate the results.
- Training curves and other statistics can be found in `metrics` for each model.

### Backbone Models

FBNet series are efficient mobile backbones discovered via neural architecture search, which are specially optimized for mobile devices. Please see details in the [paper](https://arxiv.org/pdf/2006.02049.pdf). If using our code/models in your research, please cite our paper:

```
@article{dai2020fbnetv3,
  title={FBNetV3: Joint architecture-recipe search using neural acquisition function},
  author={Dai, Xiaoliang and Wan, Alvin and Zhang, Peizhao and Wu, Bichen and He, Zijian and Wei, Zhen and Chen, Kan and Tian, Yuandong and Yu, Matthew and Vajda, Peter and others},
  journal={arXiv preprint arXiv:2006.02049},
  year={2020}
}
```



## COCO Object Detection

| name                                                         | box AP | latency | model id  | download                                                     |
| ------------------------------------------------------------ | ------ | ------- | --------- | ------------------------------------------------------------ |
| [Faster-RCNN-FBNetV3A](./configs/faster_rcnn_fbnetv3a_C4.yaml) | 22.99  | 59ms    | 246823121 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/model_0479999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/metrics.json) |
| [Faster-RCNN-FBNetV3A-dsmask](./configs/faster_rcnn_fbnetv3a_dsmask_C4.yaml) | 21.06  | 30ms    | 250414811 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250414811/model_0399999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250414811/metrics.json) |
| [Faster-RCNN-FBNetV3G-FPN](./configs/faster_rcnn_fbnetv3g_fpn.yaml) | 43.13  |         | 250356938 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250356938/model_0374999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250356938/metrics.json) |
## COCO Instance Segmentation

| name                                                         | box AP | mask AP | model id  | download                                                     |
| ------------------------------------------------------------ | ------ | ------- | --------- | ------------------------------------------------------------ |
| [Mask-RCNN-FBNetV3A](./configs/mask_rcnn_fbnetv3a_C4.yaml)   | 23.74  | 21.18   | 250355374 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250355374/model_0479999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250355374/metrics.json) |
| [Mask-RCNN-FBNetV3A-dsmask](./configs/mask_rcnn_fbnetv3a_dsmask_C4.yaml) | 21.81  | 19.76   | 250414867 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250414867/model_0399999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250414867/metrics.json) |
| [Mask-RCNN-FBNetV3G-FPN](./configs/mask_rcnn_fbnetv3g_fpn.yaml) | 43.88  | 39.25   | 250376154 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250376154/model_0404999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250376154/metrics.json) |

### COCO Person Keypoint Detection

| name                                                         | box AP | kp. AP | model id  | download                                                     |
| ------------------------------------------------------------ | ------ | ------ | --------- | ------------------------------------------------------------ |
| [Keypoint-RCNN-FBNetV3A-dsmask](./configs/keypoint_rcnn_fbnetv3a_dsmask_C4.yaml) | 31.24  | 35.56  | 250430934 | [model](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250430934/model_0389999.pth) \|[metrics](https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/250430934/metrics.json) |

