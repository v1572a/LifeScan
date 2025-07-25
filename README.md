# LifeScan

# 🧠 LifeScan: Denoising and Body Part Detection in Noisy Environments

**LifeScan** is an edge-AI deep learning pipeline designed to identify human presence in visually impaired environments such as disaster zones, rubble sites, and low-light areas. The system leverages an **autoencoder** for image denoising followed by a **CNN classifier** for body part identification. It is optimized for **on-device inference** using **ONNX quantization** and benchmarking on **Raspberry Pi 4B**.

---

## 📁 Project Structure


## Description

- **Results/**: Contains images of benchmarking of models on **Raspberry Pi 4B**.
- **src/code/**: Includes the `stats_onnx.py` script for performing inference and evaluating metrics on ONNX models.
- **src/models/**: Stores ONNX model files, split into two subdirectories:
  - **Model_without_Quantization/**: Non-quantized versions of the autoencoder and CNN models.
  - **Models_with_Quantization/**: INT8 quantized versions of the autoencoder and CNN models.
- **notebooks/**: Contains Jupyter notebooks for model quantization (`onnx_quantize.ipynb`) and graph optimization with benchmarking (`onnx_quantize_inference_graph.ipynb`).
- **README.md**: This file, providing an overview of the project structure and contents.
- **requirements.txt**: This file provides all the dependencies

## 🚀 Features

- 🔧 **Autoencoder**: Cleans noisy images with 5% salt-and-pepper noise and reduced brightness.
- 🧠 **CNN Classifier**: Identifies six body parts — `arm`, `elbow`, `face`, `foot`, `hand`, and `leg`.
- 🧊 **Quantized ONNX Models**: Lightweight and efficient models for edge deployment.
- 📉 **Benchmark Reports**: Detailed logs of RAM, CPU, and inference time in `Results/`.

## Setup and Usage


```bash
git clone https://github.com/v1572a/LifeScan.git
cd LifeScan

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python src/code/stats_onnx.py # Add paths of autoencoder and cnn classifier corrrectly
```
