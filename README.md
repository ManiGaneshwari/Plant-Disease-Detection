# Plant Disease Detection System

A machine learning-powered web application for detecting plant diseases from leaf images using deep learning and computer vision.

## 🌱 Overview

This system provides an intuitive web interface built with Streamlit for uploading plant images and receiving real-time disease predictions. The application uses a trained Convolutional Neural Network (CNN) model to classify plant diseases with confidence scores and recommendations.

## ✨ Features

- **Real-time Disease Detection**: Upload plant images and get instant predictions
- **User-friendly Interface**: Simple web-based interface with responsive design
- **High Accuracy**: CNN-based model trained on plant disease datasets
- **Local Deployment**: Runs entirely on your local machine for data privacy
- **Multiple Format Support**: Accepts JPG, JPEG, and PNG image formats
- **Confidence Scoring**: Provides prediction confidence levels
- **Mobile Responsive**: Works on desktop and mobile browsers
## 📚 Resources
* [Dataset: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [OpenCV Documentation](https://opencv.org/)

* [Kaggle Platform](https://www.kaggle.com/)
## 🔧 System Requirements

### Software Prerequisites
- **Python**: 3.7 or higher
- **pip**: Python package installer
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **GPU**: Optional (for faster inference)
- **Network**: Internet connection for initial package installation

## 📁 Project Structure

```
plant-disease-detection/
├── plant_disease_detection.py          # Main Streamlit application
├── CNN_plantdiseases_model.keras       # Trained CNN model file
├── plantdisease.png                    # Application logo/image (optional)
├── requirements.txt                    # Python dependencies
├── test_images/                        # Test images folder (optional)
│   ├── leaf.jpg                        # Sample test image 1
│   └── leaf1.jpg                       # Sample test image 2
└── README.md                           # Project documentation
```

## 🚀 Quick Start

### 1. Clone or Download Project
```bash
mkdir plant-disease-detection
cd plant-disease-detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv plant_disease_env

# Activate virtual environment
# On Windows:
plant_disease_env\Scripts\activate
# On macOS/Linux:
source plant_disease_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run plant_disease_detection.py
```

### 5. Open in Browser
Navigate to `http://localhost:8501` in your web browser.

## 📦 Installation

### Dependencies

```txt
streamlit
tensorflow
pillow
numpy
pandas
matplotlib
opencv-python
scikit-learn
```

Install dependencies:
```bash
# Try with without versions first
pip install -r requirements.txt

# If there are version conflicts, use the one with versions:
pip install -r requirements-versions.txt
```
**Note**: Use the requirements file without versions to install the latest compatible versions automatically.


### Model Setup
Ensure the trained model file `CNN_plantdiseases_model.keras` is in your project directory (The which is trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).




## 🎯 How to navigate in the web app

### Navigation
The application features a sidebar with two main sections:
1. **HOME** - Landing page with system information
2. **DISEASE RECOGNITION** - Main prediction interface

### Making Predictions

1. **Select Disease Recognition** from the sidebar dropdown
2. **Upload Plant Image** by clicking "Browse Files"
3. **View Results** including disease classification and confidence score

### Supported Image Formats
- JPG/JPEG
- PNG
- Recommended size: < 10MB


<details>
<summary><h3 style="margin: 0; display: inline;">📊 Dataset & 🏋️‍♂️ Model Training (Click to Expand)</h3></summary>


## 📁 Dataset Information

- **Name:** New Plant Diseases Dataset (Augmented)
- **Source:** [Kaggle – New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- **Classes:** 38 plant disease categories
- **Structure:**
  ```
  dataset/
    ├── train/    # Training images per class
    ├── valid/    # Validation images per class
    └── test/     # Testing images (optional)
  ```

> 💡 **Setup:** Download the dataset from Kaggle and place it in your project directory under the respective folders: `train/`, `valid/`, and `test/`.

## 🧠 Model Architecture

The model uses **Transfer Learning** based on MobileNet, a lightweight deep learning model pretrained on ImageNet.

- **Base Model:** MobileNet (with `include_top=False`)
- **Custom Layers:**
  - Global Average Pooling
  - Dropout (0.5) for regularization
  - Dense layer with ReLU activation
  - Final Dense layer with 38 softmax outputs

## 🏁 Training the Model

The training is performed in Google Colab using the notebook `plantdisease.ipynb`. Here's the actual implementation:

### ✅ 1. Setup Google Drive & Extract Dataset
```python
from google.colab import drive
drive.mount('/content/drive')

import zipfile
import os

# Extract dataset from Google Drive
zip_file_path = "/content/drive/MyDrive/archive (2).zip"
extract_dir = "/content"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
```

### 📚 2. Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
```

### 📂 3. Setup Data Paths & Generators
```python
# Dataset paths
train_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train'
valid_dir = '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid'

# Image specifications
img_size = 224
batch_size = 32

# Data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    horizontal_flip=True,
    validation_split=0.1
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.1
)

# Load data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
```

### 🧠 4. Build the Model (Sequential Architecture)
```python
# Load pre-trained MobileNet
base_model = MobileNet(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

# Build Sequential model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(train_generator.num_classes, activation='softmax')
])
```

### ⚙️ 5. Compile the Model
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 🚀 6. Train with Early Stopping
```python
# Early stopping for optimization
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=5,
    steps_per_epoch=100,
    validation_steps=50,
    callbacks=[early_stopping]
)
```

### 📊 7. Visualize Training Results
```python
# Plot training metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim(0, 1.02)
plt.show()
```

### 🧪 8. Model Evaluation
```python
# Evaluate on test set
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.0
).flow_from_directory(
    '/content/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid',
    batch_size=164,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    shuffle=False
)

model_evaluate = model.evaluate(test_generator)
print("Loss     : ", model_evaluate[0])
print("Accuracy : ", model_evaluate[1])
```

### 💾 9. Save the Trained Model
```python
model.save('CNN_plantdiseases_model.keras')
```

### ▶️ Run Training in Google Colab

1. **Upload your dataset** to Google Drive as a zip file
2. **Open the notebook** in Google Colab
3. **Run all cells** sequentially
4. **Monitor training** with the accuracy plots

**Key Features:**
- **Optimized for Colab:** Uses limited steps per epoch for faster training
- **Early Stopping:** Prevents overfitting with patience=3
- **Data Augmentation:** Horizontal flip for better generalization
- **Sequential Architecture:** Simplified model building approach

> ⏱️ **Training Time:** Approximately 2-3 hours.

</details>


<details>
<summary><h3 style="margin: 0; display: inline;">🛠️ Troubleshooting (Click to Expand)</h3></summary>

### Common Issues & Solutions

<details>
<summary><strong>🔧 Module Not Found Error</strong></summary>
<br>

If you encounter module import errors, try these solutions in order:

**Option 1: Install with specific versions**
```bash
pip install -r requirements.txt
```

**Option 2: Install without version constraints**
```bash
pip install -r requirements-no-versions.txt
```

**Option 3: Install individual packages**
```bash
pip install streamlit tensorflow pillow numpy pandas matplotlib opencv-python scikit-learn
```

**Option 4: Use virtual environment**
```bash
python -m venv plant_disease_env
source plant_disease_env/bin/activate  # On Windows: plant_disease_env\Scripts\activate
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>📁 Model Loading Error</strong></summary>
<br>

If the model fails to load, check the following:

**Verify model file exists:**
```bash
ls -la CNN_plantdiseases_model.keras
```

**Ensure correct file path:**
```python
import os
print("Current directory:", os.getcwd())
print("Model file exists:", os.path.exists("CNN_plantdiseases_model.keras"))
```

**Alternative loading methods:**
```python
# Try absolute path
model_path = os.path.abspath("CNN_plantdiseases_model.keras")
model = tf.keras.models.load_model(model_path)

# Or specify custom objects if needed
model = tf.keras.models.load_model("CNN_plantdiseases_model.keras", compile=False)
```

</details>

<details>
<summary><strong>🌐 Port Already in Use</strong></summary>
<br>


If you get a "port already in use" error when running Streamlit:

**Use different port:**
```bash
streamlit run plant_disease_detection.py --server.port 8502
```

**Alternative ports to try:**
```bash
streamlit run plant_disease_detection.py --server.port 8503
streamlit run plant_disease_detection.py --server.port 8504
streamlit run plant_disease_detection.py --server.port 8505
```

</details>

<details>
<summary><strong>🖼️ Image Upload Issues</strong></summary>
<br>

If images won't upload or process correctly:

**Check supported formats:**
- ✅ Supported: JPG, JPEG, PNG
- ❌ Not supported: GIF, BMP, TIFF, WEBP

**Verify file size:**
- Recommended: < 10MB
- Maximum: < 200MB

**Check image integrity:**
```python
from PIL import Image
try:
    img = Image.open("your_image.jpg")
    img.verify()  # Check if image is valid
    print("Image is valid")
except Exception as e:
    print(f"Image error: {e}")
```

**Common solutions:**
- Convert image to RGB format
- Resize large images before upload
- Ensure image isn't corrupted
- Try different image format

</details>

<details>
<summary><strong>🐍 Python Version Issues</strong></summary>
<br>

If you encounter Python compatibility issues:

**Check Python version:**
```bash
python --version
```

**Recommended versions:**
- Python 3.8 - 3.10 (recommended)
- Python 3.11+ may have compatibility issues with some TensorFlow versions

**If using wrong Python version:**
```bash
# Install specific Python version using pyenv
pyenv install 3.9.16
pyenv local 3.9.16

# Or use conda
conda create -n plant_disease python=3.9
conda activate plant_disease
```

</details>

<details>
<summary><strong>💾 Memory Issues</strong></summary>
<br>

If you encounter out-of-memory errors:

**For training:**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Reduce steps per epoch
steps_per_epoch = 50  # Instead of 100
```

**For inference:**
```python
# Clear memory after prediction
import gc
tf.keras.backend.clear_session()
gc.collect()
```

**System recommendations:**
- Minimum: 8GB RAM
- Recommended: 16GB+ RAM
- Use GPU if available for faster processing

</details>

</details>

<details>
<summary><h3 style="margin: 0; display: inline;">🔒 Security & Privacy (Click to Expand)</h3></summary>


- **Local Processing**: All images processed locally, not sent to external servers
- **No Data Storage**: User uploads are temporary and not permanently stored
- **File Validation**: Automatic validation of file types and sizes

**Privacy Features:**
- No user data collection
- No tracking or analytics
- Secure local environment processing
- Images deleted after processing

</details>

<details>
<summary><h3 style="margin: 0; display: inline;">🔧 Development (Click to Expand)</h3></summary>
  
### Prerequisites for Development
- Python 3.7+
- TensorFlow 2.15+
- Streamlit 1.28+

### Local Development Setup
```bash
git clone <repository-url>
cd plant-disease-detection
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run plant_disease_detection.py
```

### Development Workflow
1. **Fork the repository** and clone locally
2. **Create a virtual environment** for isolation
3. **Install dependencies** from requirements.txt
4. **Make your changes** and test thoroughly
5. **Submit a pull request** with detailed description

### Testing
```bash
# Run basic functionality tests
python -m pytest tests/

# Test the Streamlit app
streamlit run plant_disease_detection.py
```

</details>

<details>
<summary><h3 style="margin: 0; display: inline;">📊 Model Information (Click to Expand)</h3></summary>
  
### Model Architecture
- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Base Model**: MobileNet (Transfer Learning)
- **Input**: RGB images of plant leaves (224x224 pixels)
- **Output**: Disease classification with confidence scores
- **Classes**: 38 different plant disease categories

### Dataset Details
- **Source**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle
- **Size**: Augmented dataset with thousands of images
- **Format**: JPG/PNG images
- **Training Split**: 80% training, 20% validation

### Model Performance
- **Training Accuracy**: Optimized with early stopping
- **Validation**: Cross-validated on separate dataset
- **File Size**: ~15MB (optimized for deployment)
- **Inference Time**: < 2 seconds per image

</details>

<details>
<summary><h3 style="margin: 0; display: inline;">🆘 Support (Click to Expand)</h3></summary>

### Getting Help

**For technical issues:**
1. Check the [Troubleshooting section](#troubleshooting) first
2. Review common issues and solutions
3. Search existing GitHub issues
4. Create a new issue with detailed description

**For questions:**
- Review the documentation thoroughly
- Check the FAQ section
- Contact the development team via GitHub

**When reporting issues, please include:**
- Python version and OS
- Error messages (full stack trace)
- Steps to reproduce the issue
- Screenshots if applicable

### Contributing
- Fork the repository
- Create a feature branch
- Submit pull requests
- Follow coding standards

</details>


<details>
<summary><h3 style="margin: 0; display: inline;">📈 Version Information (Click to Expand)</h3></summary>

### Current Version
- **Application Version**: 1.0.0
- **Release Date**: Current
- **Status**: Stable

### Compatibility
- **Python Compatibility**: 3.7+
- **TensorFlow Version**: 2.15+
- **Streamlit Version**: 1.28+
- **Operating Systems**: Windows, macOS, Linux

### Dependencies
```txt
tensorflow>=2.15.0
streamlit>=1.28.0
pillow>=8.3.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
```

### Version History
- **v1.0.0**: Initial release with CNN model
- **Future**: Planned improvements and new features

### System Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB+
- **Storage**: 500MB free space
- **GPU**: Optional (for faster processing)

</details>

---

**Made with ❤️ for agricultural technology and plant health monitoring**


