<h1 align="center">Hand Recognition Project ✋🤖</h1>

<p align="center">
  <a href="https://github.com/Halo201417/HandRecogProject/stargazers">
    <img src="https://img.shields.io/github/stars/Halo201417/HandRecogProject.svg?colorA=363a4f&colorB=b7bdf8&style=for-the-badge" alt="Stars">
  </a>
  <a href="https://github.com/Halo201417/HandRecogProject/issues">
    <img src="https://img.shields.io/github/issues/Halo201417/HandRecogProject.svg?colorA=363a4f&colorB=f5a97f&style=for-the-badge" alt="Issues">
  </a>
  <a href="https://github.com/Halo201417/HandRecogProject/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Halo201417/HandRecogProject.svg?colorA=363a4f&colorB=a6da95&style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <b>A real-time Sign Language to text translation system, powered by Artificial Intelligence and MediaPipe.</b>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [📋 Prerequisites](#-prerequisites)
- [⚙️ Installation and Setup](#️-installation-and-setup)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create and activate the Virtual Environment (`venv`)](#2-create-and-activate-the-virtual-environment-venv)
  - [3. Install Dependencies](#3-install-dependencies)
- [🚀 System Usage](#-system-usage)
  - [1. Data Collection](#1-data-collection)
  - [2. Model Training (`src/train_model_lstm.py`)](#2-model-training-srctrain_model_lstmpy)
  - [3. Real-Time Translation (`src/main.py`)](#3-real-time-translation-srcmainpy)
- [📁 Code Structure](#-code-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## 📋 Prerequisites

This project has been developed, optimized, and thoroughly tested on **Debian 12 (Bookworm)**, and features cross-support for Edge boards like the **Raspberry Pi**.

* **Operating System:** Debian 12 or derivative Linux distributions (or Raspberry Pi OS).
* **Python:** Python 3.11 (default native version on Debian 12).
* **Hardware:** * A functional webcam.
    * Processor compatible with x86_64 (PC) or aarch64 (Raspberry Pi ARM).

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## ⚙️ Installation and Setup

Follow these steps to configure the environment in an isolated and secure way using virtual environments.

### 1. Clone the repository
Open a terminal and download the project to your local computer:
```bash
git clone https://github.com/Halo201417/HandRecogProject.git
```

### 2. Create and activate the Virtual Environment (`venv`)
Isolate dependencies to avoid conflicts with your global Linux libraries:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
With the `(venv)` environment activated, install exactly the tested required versions for maximum compatibility:
```bash
pip install -r requirements.txt
```
> **💡 Cross-platform Note:** The `requirements.txt` file is purged of proprietary drivers to ensure that the installation of heavy libraries (TensorFlow/Keras, OpenCV) runs without errors regardless of the CPU you use.

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## 🚀 System Usage

The workflow is modular and divided into 3 phases (always run from the project root):

### 1. Data Collection
Create your own dataset by recording gestures with the camera in real-time. The project uses two distinct collection engines depending on the gesture type:

**For Static Gestures (Standard Alphabet):**
* **Command:** `python src/data_collection.py`
* Press the keyboard key corresponding to the letter you are signing to capture its static coordinate frame.

**For Dynamic Gestures & Commands (Temporal Sequences):**
* **Command:** `python src/data_collection_seq.py`
* Use the specific bound keys to record 3D movement sequences:
  * Press `Z` to record the dynamic letter **Z**.
  * Press `C` to record the **CONFIRM** command.
  * Press `D` to record the **DELETE** command.
  * Press `F` to record the **CLEAR** (Final) command.

> **💡 Note:** In both data collection scripts, press the `ESC` key to safely close the camera and save your progress.

### 2. Model Training (`src/train_model_lstm.py`)
Train the LSTM neural network combining both the static and dynamic data collected (`X_data.npy` and `y_data.npy`):
* **Command:** `python src/train_model_lstm.py`
* The system will balance the data and generate a `hand_model_lstm.h5` file and a visual training graph.

### 3. Real-Time Translation (`src/main.py`)
Open the webcam and classify your movements instantly:
* **Command:** `python src/main.py`
* *Built-in NLP Logic:* Make signs to translate letters. Use the dynamic gestures (`CONFIRM`, `DELETE`, `CLEAR`) to manage the text buffer and form complete words.

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## 📁 Code Structure

* `src/main.py` - Main translation application and cross-platform logic (PC/Raspberry Pi).
* `src/train_model_lstm.py` - Neural network training architecture (TimeDistributed + LSTM).
* `src/data_collection_seq.py` - Temporal capture engine using OpenCV.
* `src/data_collection.py` - Temporal capture engine using OpenCV.
* `src/detector.py` - Optimized wrapper for MediaPipe bone mapping AI.
* `requirements.txt` - Production environment dependency list.

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="line">
</p>

## 🤝 Contributing

Contributions, issues, and pull requests are welcome! 
Feel free to check the [issues page](https://github.com/Halo201417/HandRecogProject/issues).

## 📄 License

This project is licensed under the **MIT** license. You can freely use, modify, and distribute it.