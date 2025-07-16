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
# Try with specific versions first
pip install -r requirements.txt

# If version conflicts occur, use:
pip install -r requirements-no-versions.txt
```

### 4. Run the Application
```bash
streamlit run plant_disease_detection.py
```

### 5. Open in Browser
Navigate to `http://localhost:8501` in your web browser.

## 📦 Installation

### Dependencies

#### Option 1: With Specific Versions (Recommended)
Create a `requirements.txt` file with:

```txt
streamlit==1.28.0
tensorflow==2.15.0
pillow==10.0.0
numpy==1.24.0
pandas==2.0.0
matplotlib==3.7.0
opencv-python==4.8.0.74
scikit-learn==1.3.0
```

#### Option 2: Without Versions (If Version Conflicts Occur)
If you encounter version compatibility issues, create a `requirements-no-versions.txt` file with:

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
# Try with specific versions first
pip install -r requirements.txt

# If there are version conflicts, use:
pip install -r requirements-no-versions.txt
```

**Note**: If you encounter version-related errors with the specific versions, use the requirements file without versions to install the latest compatible versions automatically.

### Model Setup
Ensure the trained model file `CNN_plantdiseases_model.keras` is in your project directory. This model should be trained on the [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).

Update the model path in the main application file:

```python
model = tf.keras.models.load_model(r"./CNN_plantdiseases_model.keras")
```

**Note**: The model file is not included in this repository due to size constraints. You need to train your own model using the provided dataset or obtain a pre-trained model.

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


## 🧠 Training Your Own Model

If you don't have a pre-trained model, you can train one using Google Colab. Follow these steps:

### Step 1: Setup Google Colab Environment
- Open [Google Colab](https://colab.research.google.com/)
- Upload the notebook `plantdisease.ipynb` to Google Drive
- Right-click → "Open With" → "Google Colab"

### Step 2 : Mount Google Drive
The first cell in the notebook runs:

```python
from google.colab import drive
drive.mount('/content/drive')
```

🔑 This step allows access to your Drive files.



### Step 2: Install Required Libraries
```python
# Install required libraries
!pip install tensorflow==2.15.0
!pip install opencv-python
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install pillow


```

### Step 3: Extract Dataset from Drive
Make sure your dataset ZIP (archive (2).zip) is in your Google Drive (MyDrive). The notebook will automatically extract it:

```python
zip_file_path = "/content/drive/MyDrive/archive (2).zip"
extract_dir = "/content"

# Extract ZIP contents to /content

```



### Step 4: Data Preprocessing
The dataset is automatically unzipped and organized. The notebook includes image augmentation, resizing, and generator setup:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

```
Target size: (224, 224)

Batch size: 32

Classes are inferred automatically

### Step 5: Build and Train CNN Model
The model is based on MobileNet, using transfer learning with custom dense layers:
```python
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout


```

### Step 6: Save and Download Model
```python
# Save the final model
model.save('CNN_plantdiseases_model.keras')

# Download the trained model
from google.colab import files
files.download('CNN_plantdiseases_model.keras')
```

### Step 7: Use the Downloaded Model
1. Download the `CNN_plantdiseases_model.keras` file to your local machine
2. Place it in your project directory
3. Run the Streamlit application as described in the Quick Start section

**Training Time**: Approximately 2-4 hours depending on epochs and GPU allocation.


## 🛠️ Troubleshooting

### Common Issues

**Module Not Found Error:**
```bash
# Try with specific versions first
pip install -r requirements.txt

# If version conflicts occur, use:
pip install -r requirements-no-versions.txt

# Or install individual packages:
pip install streamlit tensorflow pillow numpy pandas matplotlib opencv-python scikit-learn
```

**Model Loading Error:**
- Verify `CNN_plantdiseases_model.keras` exists in project directory
- Check file permissions
- Ensure correct file path in code

**Port Already in Use:**
```bash
streamlit run plant_disease_detection.py --server.port 8502
```

**Image Upload Issues:**
- Ensure supported format (JPG, PNG)
- Check file size (< 10MB recommended)
- Verify image isn't corrupted

## 🔒 Security & Privacy

- **Local Processing**: All images processed locally, not sent to external servers
- **No Data Storage**: User uploads are temporary and not permanently stored
- **File Validation**: Automatic validation of file types and sizes



## 🔧 Development

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

## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: RGB images of plant leaves
- **Output**: Disease classification with confidence scores
- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
- Check the troubleshooting section
- Review common issues and solutions
- Contact the development team



## 📈 Version Information

- **Application Version**: 1.0.0
- **Python Compatibility**: 3.7+
- **TensorFlow Version**: 2.15+
- **Streamlit Version**: 1.28+

---

**Made with ❤️ for agricultural technology and plant health monitoring**
