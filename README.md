# 🧿 Vision AI: Beyond the Image, Into the Insight

This is an advanced, multi-functional computer vision application built with Python and Streamlit. It provides a user-friendly web interface to perform a wide array of AI-powered tasks on any uploaded image. The application leverages powerful, pre-trained models from Hugging Face and other open-source libraries to deliver state-of-the-art results without needing to train models from scratch.

## ✨ Features

This application combines five key computer vision tasks into a single, cohesive interface:

### 📝 Image Captioning
- Automatically generates a descriptive sentence about the contents of the image
- Allows for "controllable captioning" by letting the user provide a starting prompt to guide the output

### ❓ Visual Question Answering (VQA)
- Lets users ask questions about the image in natural language (e.g., "What color is the car?")
- The AI analyzes the image to provide a text-based answer

### 🎯 Object Detection
- Identifies and draws bounding boxes around multiple objects in an image
- Displays a detailed table listing each detected object and its confidence score

### 🖼️ Image Segmentation & Portrait Mode
- Performs semantic segmentation to understand the exact shape and boundaries of objects
- Applies a "Portrait Mode" effect by blurring the background and keeping the main subject (typically a person) in sharp focus
- Includes the ability to remove the blur effect and revert to the original segmentation mask

### 📖 Optical Character Recognition (OCR)
- Detects and extracts any text present in the image
- Includes image pre-processing (grayscale conversion, contrast enhancement) to improve accuracy on difficult text, such as logos
- Draws bounding boxes around the detected text on the original image

## 🚀 How to Run the Application

Follow these steps to get the application running on your local machine.

### 1. Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### 2. Installation
Open your terminal or command prompt and install all the required libraries by running the following command:

```bash
pip install streamlit pandas torch torchvision transformers easyocr numpy Pillow
```

### 3. Running the App
Once the installation is complete, navigate to the directory where you saved `app.py` and run this command:

```bash
streamlit run app.py
```

A new tab should automatically open in your web browser with the Vision AI application running.

**Note:** The first time you run the app, it will take several minutes to download the pre-trained AI models. This is a one-time process. Subsequent startups will be much faster as the models will be loaded from the cache.

## 🤖 Models Used

This application relies on the following pre-trained models:

- **Image Captioning:** `Salesforce/blip-image-captioning-large`
- **Visual Question Answering:** `Salesforce/blip-vqa-base`
- **Object Detection:** `facebook/detr-resnet-50`
- **Image Segmentation:** `nvidia/segformer-b5-finetuned-ade-640-640`
- **Optical Character Recognition:** `easyocr` library

## 📂 File Structure

```
vision-ai/
├── app.py              # Main Python script with application logic and Streamlit UI
├── README.md           # Project documentation (this file)
└── requirements.txt    # Python dependencies (optional)
```

## 📋 Requirements

If you prefer to use a `requirements.txt` file, create one with the following content:

```
streamlit
pandas
torch
torchvision
transformers
easyocr
numpy
Pillow
```

Then install dependencies using:
```bash
pip install -r requirements.txt
```

## 🖥️ Usage

1. **Upload an Image:** Click on the file uploader and select an image (supports JPG, PNG, JPEG formats)

2. **Choose a Task:** Select from the available computer vision tasks:
   - Image Captioning
   - Visual Question Answering
   - Object Detection
   - Image Segmentation & Portrait Mode
   - Optical Character Recognition

3. **Interact with Results:** Each task provides different interaction options:
   - For VQA: Type your question about the image
   - For Captioning: Optionally provide a starting prompt
   - For Portrait Mode: Toggle blur effects on/off
   - For OCR: View extracted text and bounding boxes

## 🔧 Troubleshooting

### Common Issues:

**Slow First Run:** The application downloads large pre-trained models on first use. This is normal and only happens once.

**Memory Issues:** If you encounter out-of-memory errors, try:
- Using smaller images
- Closing other applications
- Ensuring you have at least 4GB of available RAM

**Installation Problems:** Make sure you have the correct Python version (3.7+) and try updating pip:
```bash
pip install --upgrade pip
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Hugging Face for providing pre-trained models
- Streamlit for the amazing web app framework
- NVIDIA, Salesforce, and Facebook/Meta for their open-source computer vision models
- EasyOCR team for the OCR capabilities

--

**Happy Vision AI exploration! 🚀👁️**
