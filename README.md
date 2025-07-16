# 🌿 Plant Disease Detection using CNN

A simple web application built with **Streamlit** that uses a trained **Convolutional Neural Network (CNN)** model to detect plant diseases from leaf images.
## 🌱 Overview

This system provides an intuitive web interface built with Streamlit for uploading plant images and receiving real-time disease predictions. The application uses a trained Convolutional Neural Network (CNN) model to classify plant diseases with confidence scores and recommendations.

## 📚 Resources
* [Dataset: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* [Streamlit Documentation](https://docs.streamlit.io/)
* [TensorFlow Documentation](https://www.tensorflow.org/)
* [OpenCV Documentation](https://opencv.org/)

* [Kaggle Platform](https://www.kaggle.com/)

## 📈 Version Information

* **Application Version**: 1.0.0
* **Python Compatibility**: 3.7+
* **TensorFlow Version**: 2.15+
* **Streamlit Version**: 1.28+

---

## ✨ Features

- **Image Upload**: Upload leaf images in common formats (JPG, PNG, JPEG)
- **Disease Detection**: Automatically detect plant diseases using a trained CNN model
- **Real-time Results**: Get instant predictions with confidence scores
- **User-friendly Interface**: Clean and intuitive Streamlit web interface
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🔧 Prerequisites

Before running the application, ensure you have:

- Python 3.7 or higher
- pip package manager
- A web browser














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
Ensure the trained model file `CNN_plantdiseases_model.keras` is in your project directory. Update the model path in the main application file:

```python
model = tf.keras.models.load_model(r"./CNN_plantdiseases_model.keras")
```

## 🎯 Usage

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

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)

## 📈 Version Information

- **Application Version**: 1.0.0
- **Python Compatibility**: 3.7+
- **TensorFlow Version**: 2.15+
- **Streamlit Version**: 1.28+

---

**Made with ❤️ for agricultural technology and plant health monitoring**
