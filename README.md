# Photonest
a web based application that will automatically detect the faces in the user uploaded images using MTCNN then does feature extraction using MobileNetV3 and uses LDA for cross validation. Then organizes the similar images using SOM based ANN.
# Photonest – AI Powered Face-Based Image Clustering Web App

## Overview

Photonest is an AI-powered web application that automatically detects faces in user-uploaded images and organizes visually similar images into clusters.

The system uses advanced Computer Vision and Deep Learning techniques including:

- MTCNN for face detection
- MobileNetV3 for feature extraction
- LDA for dimensionality reduction and cross validation
- SOM-based Artificial Neural Networks for image clustering

Photonest helps automate image organization and clustering, reducing manual effort while improving efficiency and scalability.

---

# Features

- Upload images through a web-based interface
- Automatic face detection using MTCNN
- Deep feature extraction using MobileNetV3
- Dimensionality reduction using LDA
- Clustering using Self-Organizing Maps (SOM)
- Automatic grouping of similar images
- Efficient processing of large image datasets
- Interactive and scalable image organization pipeline

---

# System Workflow

```text
User Uploaded Images
          │
          ▼
   Face Detection
      (MTCNN)
          │
          ▼
 Feature Extraction
   (MobileNetV3)
          │
          ▼
Dimensionality Reduction
        (LDA)
          │
          ▼
 Image Clustering
   (SOM-based ANN)
          │
          ▼
 Organized Similar Images
```

---

# Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| Flask / Streamlit | Web Application Framework |
| OpenCV | Image Processing |
| MTCNN | Face Detection |
| MobileNetV3 | Feature Extraction |
| Scikit-learn | LDA Implementation |
| MiniSom / SOMPY | SOM Clustering |
| NumPy & Pandas | Data Processing |
| Matplotlib / Plotly | Visualization |

---

# Methodology

## 1. Face Detection using MTCNN

The uploaded images are first processed using Multi-task Cascaded Convolutional Networks (MTCNN) to accurately detect and crop faces.

### Advantages:
- High detection accuracy
- Handles multiple faces
- Robust to lighting and pose variations

---

## 2. Feature Extraction using MobileNetV3

Detected faces are passed through MobileNetV3 to generate high-dimensional facial embeddings representing unique facial characteristics.

### Benefits:
- Lightweight architecture
- Fast inference
- Efficient feature representation

---

## 3. LDA for Cross Validation and Dimensionality Reduction

Linear Discriminant Analysis (LDA) is used to:
- Reduce feature dimensions
- Improve feature separability
- Validate extracted feature consistency

This improves clustering performance and computational efficiency.

---

## 4. SOM-Based Image Clustering

A Self-Organizing Map (SOM), an unsupervised Artificial Neural Network, is used to cluster similar facial embeddings.

### SOM Advantages:
- Preserves topological relationships
- Groups visually similar faces
- Works without labeled data

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/photonest.git
cd photonest
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

## Streamlit

```bash
streamlit run app.py
```

## Flask

```bash
python app.py
```

---

# Project Structure

```text
photonest/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── mobilenetv3.py
│   └── som_model.py
│
├── utils/
│   ├── face_detection.py
│   ├── feature_extraction.py
│   ├── lda_processing.py
│   └── clustering.py
│
├── uploads/
├── clustered_images/
└── static/
```

---

# Results

Photonest successfully:
- Detects faces from uploaded images
- Extracts meaningful facial embeddings
- Organizes visually similar images automatically
- Reduces manual image sorting effort
- Handles large-scale image datasets efficiently

---

# Future Enhancements

- Real-time webcam integration
- Face recognition support
- Cloud storage integration
- Advanced clustering visualization
- Multi-user support
- Hybrid clustering techniques

---

# Applications

- Smart photo gallery systems
- Digital media management
- Surveillance image organization
- AI-based album generation
- Social media image clustering

---

# Conclusion

Photonest demonstrates the integration of Deep Learning and Unsupervised Neural Networks for intelligent image organization. By combining MTCNN, MobileNetV3, LDA, and SOM-based clustering, the system provides an efficient and scalable solution for automatic face-based image grouping.

---

# Author

Photonest Project  
AI/ML and Computer Vision Based Image Clustering System
