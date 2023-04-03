# Getting Started with D2Go

This document provides a brief intro of the usage of builtin command-line tools in d2go.

For a tutorial that involves coding with the API, see our [Jupyter Notebook](./d2go_beginner.ipynb) which covers 1). how to run inference with an existing model, 2). how to train a builtin model on a custom dataset, and 3). how to apply quantization to the model for int8 deployment.

## Inference Demo with Pre-trained Models

- Choose a model from [model_zoo](https://github.com/facebookresearch/d2go/blob/master/MODEL_ZOO.md), e.g. `faster_rcnn_fbnetv3a_C4.yaml`.
- Use the provided `demo.py` to try demo on an input image:

```bash
cd demo/
python demo.py --config-file faster_rcnn_fbnetv3a_C4.yaml --input input1.jpg --output output1.jpg
```

- To run on a video, replace the `--input files` with `--video-input video.mp4`

## Training & Evaluation

D2Go is built on top of detectron2 toolkit, please follow the [instructions](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) on detectron2 to setup the builtin datasets before training.

- To train a model:

```bash
d2go.train_net --config-file ./configs/faster_rcnn_fbnetv3a_C4.yaml
```

- To evaluate a model checkpoint:

```bash
d2go.train_net --config-file ./configs/faster_rcnn_fbnetv3a_C4.yaml --eval-only \
MODEL.WEIGHTS https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/model_0479999.pth
```

(change the URL to a local path, if evaluating local models)

## Export to Torchscript & Int8 Model

- Export to Torchscript model:

```bash
d2go.exporter --config-file configs/faster_rcnn_fbnetv3a_C4.yaml \
--predictor-types torchscript --output-dir ./ \
MODEL.WEIGHTS https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/model_0479999.pth
```

- Export to Int8 model (using post-training quantization):

```bash
d2go.exporter --config-file configs/faster_rcnn_fbnetv3a_C4.yaml \
--predictor-type torchscript_int8 --output-dir ./ \
MODEL.WEIGHTS https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/model_0479999.pth
```

## Quantization-aware Training

The previous method exports int8 models using post-training quantization, which is easy-to-use but may suffers accuracy drop. Quantization aware training emulates inference-time quantization during the training, so that the resulting lower-precision model can benefit during deployment.

To apply quantization-aware training, we need to resume from a pretrained checkpoint:

```bash
d2go.train_net --config-file configs/qat_faster_rcnn_fbnetv3a_C4.yaml \
MODEL.WEIGHTS https://mobile-cv.s3-us-west-2.amazonaws.com/d2go/models/246823121/model_0479999.pth
```

Please see the [config file](../configs/qat_faster_rcnn_fbnetv3a_C4.yaml) for relevant hyper-params.
