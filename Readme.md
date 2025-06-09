# TensorFlow Object Detection Model Training and Deployment

This project provides a comprehensive guide and implementation for training a custom object detection model using the TensorFlow Object Detection (TFOD) API on a Windows system. The model is trained to detect specific objects related to safety equipment and fall detection, such as hardhats, gloves, goggles, safety vests, ladders, and more. The pipeline includes setup, training, evaluation, and deployment for real-time detection, as well as conversion to TensorFlow.js (TFJS) and TensorFlow Lite (TFLite) formats for web and mobile applications.in order to get the evaluation score run accuracy.py file and the images also included .

---

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Directory Structure](#directory-structure)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Real-Time Detection](#real-time-detection)
- [Model Export](#model-export)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The project uses the TensorFlow Object Detection API to train a custom SSD MobileNet V2 FPNLite model for detecting 14 classes related to safety equipment and fall detection. The implementation is based on the TFOD API tutorial and includes:

- Setting up a Conda environment with TensorFlow 2.10.0 (CPU mode).
- Downloading a pre-trained model from the TensorFlow Model Zoo.
- Preparing a custom dataset with TFRecords and a label map.
- Configuring and training the model.
- Evaluating the model using COCO metrics.
- Performing real-time detection with a webcam.
- Exporting the model to TFJS and TFLite formats.

The Jupyter notebook `main.ipynb` orchestrates the entire pipeline, from setup to deployment.

---

## Prerequisites

### Operating System

- Windows (tested on Windows 10/11)

### Hardware

- CPU (GPU optional for faster training; requires CUDA 11.2 and cuDNN 8.1)
- At least 8 GB RAM
- Webcam for real-time detection

### Software

- Anaconda or Miniconda
- Python 3.8.x (3.8.20 recommended)
- Git
- Jupyter Notebook

### Dataset

- Images with annotations (XML format, PASCAL VOC) split into train and test folders.
- Example dataset structure:
  ```
  Tensorflow/workspace/images/train
  Tensorflow/workspace/images/test
  ```

---

## Setup Instructions

### Install Anaconda/Miniconda

1. Download and install from [Anaconda's website](https://www.anaconda.com/).
2. Verify installation:
   ```bash
   conda --version
   ```

### Create and Activate Conda Environment

```bash
conda create -n tfod_env python=3.8
conda activate tfod_env
```

### Install Dependencies

Install TensorFlow and other required packages:

```bash
pip install tensorflow==2.10.0 tensorflow-io==0.27.0 tensorflow-io-gcs-filesystem==0.27.0 tensorflow-datasets==4.9.0 array-record==0.4.0 tf-models-official==2.10.1 numpy==1.22.4 matplotlib==3.5.3 protobuf==3.19.4 tf-slim==1.1.0 pillow==9.2.0 cython lxml contextlib2 pycocotools-windows jupyter wget
```

### Clone TensorFlow Models Repository

```bash
git clone https://github.com/tensorflow/models Tensorflow/models
```

### Install Protobuf

1. Download `protoc-3.15.6-win64.zip` from [Protobuf releases](https://github.com/protocolbuffers/protobuf/releases).
2. Extract to `Tensorflow/protoc`.
3. Add to PATH:
   ```bash
   set PATH=%PATH%;D:\path\to\Tensorflow\protoc\bin
   ```

### Compile Protobuf Files

```bash
cd Tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
```

### Install TFOD API

```bash
copy object_detection\packages\tf2\setup.py setup.py
python setup.py build
python setup.py install
cd slim
pip install -e .
```

### Verify Installation

Run the verification script:

```bash
python object_detection/builders/model_builder_tf2_test.py
```

Expected output: `Ran 24 tests in X.XXXs OK (skipped=1).`

---

## Directory Structure

The project uses a structured directory layout:

```
Tensorflow/
├── workspace/
│   ├── annotations/
│   │   ├── label_map.pbtxt
│   │   ├── train.record
│   │   ├── test.record
│   ├── images/
│   │   ├── train/
│   │   ├── test/
│   │   ├── val/
│   ├── models/
│   │   ├── my_ssd_mobnet/
│   │   │   ├── pipeline.config
│   │   │   ├── checkpoint/
│   │   │   ├── export/
│   │   │   ├── tfjsexport/
│   │   │   ├── tfliteexport/
│   ├── pre-trained-models/
│       ├── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
│           ├── checkpoint/
│           ├── saved_model/
│           ├── pipeline.config
├── scripts/
│   ├── generate_tfrecord.py
├── models/
│   ├── research/
│       ├── object_detection/
│       ├── slim/
├── protoc/
│   ├── bin/
│   ├── include/
main.ipynb
```

---

## Step-by-Step Implementation

The `main.ipynb` notebook guides you through the following steps:

1. **Setup Paths**: Define paths and create directories.
2. **Download Pre-trained Model**: Download and extract `ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8`.
3. **Create Label Map**: Define 14 classes and save `label_map.pbtxt`.
4. **Create TFRecords**: Generate `train.record` and `test.record`.
5. **Copy Model Config**: Copy `pipeline.config` to the custom model directory.
6. **Update Config for Transfer Learning**: Modify `pipeline.config` for custom training.
7. **Train the Model**: Run training for 2000 steps.
8. **Evaluate the Model**: Evaluate using COCO metrics.
9. **Real-Time Detection**: Perform detection using a webcam.
10. **Model Export**: Export to frozen graph, TFJS, and TFLite formats.

---

## Training the Model

To train the model:

```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=2000
```

---

## Evaluating the Model

To evaluate:

```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --checkpoint_dir=Tensorflow/workspace/models/my_ssd_mobnet --logtostderr
```

---

## Real-Time Detection

The notebook includes code for real-time detection using a webcam. Press `q` to exit the detection window.

---

## Model Export

- **Frozen Graph**: For TensorFlow environments.
- **TFJS**: For web-based applications.
- **TFLite**: For mobile/embedded devices.

---

## Troubleshooting

- **Verification Script Fails**: Check `PYTHONPATH`.
- **TFRecord Generation Errors**: Ensure XML annotations match `label_map.pbtxt`.
- **Training/Evaluation Errors**: Confirm `pipeline.config` paths are correct.
- **Real-Time Detection Issues**: Ensure webcam is connected.

---

