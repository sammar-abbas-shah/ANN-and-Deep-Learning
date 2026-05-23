# 🧠 ANN and Deep Learning — Fifth Semester Academic Repository

> **By [Sammar Abbas](https://www.linkedin.com/in/sammar-abbas-372359289)**  
> 📧 [sammarabbas0619@gmail.com](mailto:sammarabbas0619@gmail.com)  
> 🔗 [LinkedIn](https://www.linkedin.com/in/sammar-abbas-372359289) | 💻 [GitHub](https://github.com/sammar-abbas-shah/ANN-and-Deep-Learning)

---

## 📌 About This Repository

This repository contains all the work completed by **Sammar Abbas** during the **Artificial Neural Networks (ANN) and Deep Learning** course in her **Fifth Semester**. It covers a broad range of deep learning concepts and architectures — from foundational perceptrons to Generative Adversarial Networks — implemented through hands-on Jupyter Notebooks.

The repository is organized into two primary sections: a **Project** (the core deliverable) and **CourseWork** (lab and assignment notebooks), supported by a rich image **dataset** and a supplementary **csv** folder.

---

## 📁 Repository Structure

```
ANN-and-Deep-Learning/
│
├── Project/                  ← 🌟 Main semester project (gesture recognition)
│   ├── CNN.ipynb             ← Convolutional Neural Network model
│   ├── LeNet.ipynb           ← LeNet-5 architecture implementation
│   ├── GAN_project.ipynb     ← Generative Adversarial Network for gesture data
│   ├── Autoencode_Project.ipynb ← Autoencoder for image reconstruction
│   ├── rough_ann.ipynb       ← ANN baseline model for gesture classification
│   ├── rough_32.ipynb        ← ANN experiments on 32x32 grayscale data
│   ├── resize.ipynb          ← Image preprocessing & resizing pipeline
│   └── generated_images/     ← GAN-generated gesture images (epoch-by-epoch)
│
├── dataset/                  ← 🌟 Gesture image dataset (linked to Project)
│   ├── Gesture_0/            ← 1000 images of gesture class 0
│   ├── Gesture_2/            ← 1000 images of gesture class 2
│   ├── Gesture_13/           ← 1000 images of gesture class 13
│   └── requirements.txt      ← Script to convert images → CSV formats
│
├── CourseWork/               ← Lab exercises and assignments
│   ├── Lab1_perceptron.ipynb
│   ├── Lab2.ipynb
│   ├── Lab3_Activation.ipynb
│   ├── Lab4_Images.ipynb
│   ├── Lab5_ANN.ipynb
│   ├── Lab6_.ipynb
│   ├── Final.ipynb
│   ├── Final_with_graph.ipynb
│   ├── Autoencoder.ipynb
│   ├── DenoisingAutoencoder.ipynb
│   ├── GANs.ipynb
│   ├── GNN.ipynb
│   ├── GNN_ASS.ipynb
│   ├── ResNet.ipynb
│   ├── SOM.ipynb / SOM2.ipynb
│   ├── Audio.ipynb / Audio_Model.ipynb
│   ├── minisom.ipynb
│   └── ann_dltassk.ipynb
│
├── csv/
│   └── heart.csv             ← Heart disease dataset (supplementary)
│
└── .gitignore                ← Excludes large generated CSV files
```

---

## 🌟 Main Project: Hand Gesture Recognition

> The `Project/` folder is the **heart of this repository** and directly depends on the `dataset/` folder. Together they form a complete end-to-end deep learning pipeline for **hand gesture classification**.

### 🎯 Objective

Build and compare multiple deep learning architectures to classify hand gesture images into one of three categories (Gesture_0, Gesture_2, Gesture_13).

---

### 🖼️ Dataset (`dataset/`)

The `dataset/` folder contains a custom hand gesture image dataset collected across **three gesture classes**, with approximately **1,000 JPEG images per class** (~3,000 images total).

| Folder | Class | Image Count |
|--------|-------|-------------|
| `Gesture_0/` | Gesture 0 | ~1,000 |
| `Gesture_2/` | Gesture 2 | ~1,000 |
| `Gesture_13/` | Gesture 13 | ~1,000 |

The images are processed and exported into CSV files in three resolutions for use across different model architectures:

| CSV Format | Resolution | Channels | Pixels per Image |
|---|---|---|---|
| `GestureDataset.csv` | 28×28 | RGB (3) | 2,352 |
| `gestures_32x32x1.csv` | 32×32 | Grayscale (1) | 1,024 |
| `GestureDataset_110x110x3.csv` | 110×110 | RGB (3) | 36,300 |

The `dataset/requirements.txt` file contains a **complete Python script** using OpenCV and Pandas to automate the image-to-CSV conversion pipeline (loading, resizing, flattening, and labeling all images).

> ⚠️ *Note: The large CSV files are excluded from the repo via `.gitignore` to keep it lightweight. Run the conversion script yourself after cloning.*

---

### 🔬 Project Notebooks (`Project/`)

#### 1. `rough_ann.ipynb` — ANN Baseline
A standard fully-connected Artificial Neural Network trained on the 110×110×3 flattened gesture data. Architecture: Dense(512, ReLU) → Dense(256, ReLU) → Dense(3, Softmax). Trained for 25 epochs using SGD with categorical cross-entropy loss.

#### 2. `CNN.ipynb` — Convolutional Neural Network
A custom CNN trained on 110×110×3 gesture images. Uses Conv2D → BatchNorm → MaxPooling → Dropout blocks, compiled with the Adam optimizer. Trained for 20 epochs with validation tracking and accuracy/loss plots.

#### 3. `LeNet.ipynb` — LeNet-5 Architecture
Classic LeNet-5 architecture adapted for gesture classification on 28×28×3 images. Uses AveragePooling and Tanh activations to stay true to the original 1998 design, while testing it on the custom gesture dataset.

#### 4. `GAN_project.ipynb` — Generative Adversarial Network
A full GAN implementation trained on the gesture CSV data (110×110×3). Includes a Generator (Dense → BatchNorm → LeakyReLU → Reshape → Conv2DTranspose blocks) and a Discriminator. Generates new gesture-like images from random noise. Epoch-by-epoch generated images are saved in `generated_images/`.

#### 5. `Autoencode_Project.ipynb` — Autoencoder
An unsupervised autoencoder that compresses gesture images (110×110×3) into a 20-dimensional latent space and reconstructs them. Encoder: Flatten → Dense(100) → Dense(20). Decoder: Dense(100) → Dense(36300) → Reshape. Useful for learning compressed representations of gesture images.

#### 6. `rough_32.ipynb` — ANN on 32×32 Grayscale
ANN experiments conducted on the smaller 32×32 grayscale version of the dataset. Used for quick experimentation and comparison with higher-resolution models.

#### 7. `resize.ipynb` — Preprocessing Pipeline
Image resizing and format conversion utilities. Handles transforming raw gesture images into the various CSV formats needed by the different models.

---

## 📚 CourseWork Notebooks

The `CourseWork/` folder contains lab exercises and assignments covering all major deep learning concepts taught during the semester:

| Notebook | Topic |
|---|---|
| `Lab1_perceptron.ipynb` | Perceptron implementation from scratch |
| `Lab2.ipynb` | Multi-layer networks basics |
| `Lab3_Activation.ipynb` | Activation functions (ReLU, Sigmoid, Tanh, etc.) |
| `Lab4_Images.ipynb` | Image loading and preprocessing with neural networks |
| `Lab5_ANN.ipynb` | Full ANN architecture and training |
| `Lab6_.ipynb` | Advanced ANN topics |
| `Final.ipynb` | Final lab implementation |
| `Final_with_graph.ipynb` | Final implementation with training visualization |
| `Autoencoder.ipynb` | Autoencoder architecture |
| `DenoisingAutoencoder.ipynb` | Denoising autoencoder with noise injection |
| `GANs.ipynb` | Generative Adversarial Networks theory & practice |
| `GNN.ipynb` / `GNN_ASS.ipynb` | Graph Neural Networks |
| `ResNet.ipynb` | Residual Networks (ResNet) architecture |
| `SOM.ipynb` / `SOM2.ipynb` | Self-Organizing Maps (unsupervised clustering) |
| `minisom.ipynb` | MiniSom library experiments |
| `Audio.ipynb` / `Audio_Model.ipynb` | Audio classification with deep learning |
| `ann_dltassk.ipynb` | ANN task/assignment |

---

## 📊 Supplementary Data (`csv/`)

The `csv/` folder contains `heart.csv` — a heart disease dataset with features like age, gender, cholesterol, blood pressure, and exercise-induced angina — used for supplementary ANN classification experiments outside the main gesture project.

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Deep Learning:** TensorFlow, Keras
- **Data Processing:** NumPy, Pandas, OpenCV (cv2)
- **Visualization:** Matplotlib
- **ML Utilities:** Scikit-learn (train/test split, label encoding, scaling)
- **Notebooks:** Jupyter Notebook

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/sammar-abbas-shah/ANN-and-Deep-Learning.git
cd ANN-and-Deep-Learning

# 2. Install dependencies
pip install tensorflow keras numpy pandas opencv-python matplotlib scikit-learn tqdm

# 3. Convert dataset images to CSV
#    Edit dataset/requirements.txt with your dataset path, then run the script

# 4. Open any notebook
jupyter notebook Project/CNN.ipynb
```

> Make sure to update the dataset CSV paths inside each notebook from the original Windows paths (`D:\ANN and DL\...`) to your local paths before running.

---

## 👩‍💻 Author

**Sammar Abbas**  
Fifth Semester — Artificial Neural Networks & Deep Learning

📧 Email: [sammarabbas0619@gmail.com](mailto:sammarabbas0619@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/sammar-abbas-372359289](https://www.linkedin.com/in/sammar-abbas-372359289)  
💻 GitHub: [github.com/sammar-abbas-shah](https://github.com/sammar-abbas-shah)

---

*This repository represents semester-long academic work exploring the theory and hands-on implementation of Artificial Neural Networks and Deep Learning architectures.*
