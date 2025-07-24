🧿 Vision AI: Beyond the Image, Into the Insight
This is an advanced, multi-functional computer vision application built with Python and Streamlit. It provides a user-friendly web interface to perform a wide array of AI-powered tasks on any uploaded image. The application leverages powerful, pre-trained models from Hugging Face and other open-source libraries to deliver state-of-the-art results without needing to train models from scratch.

✨ Features
This application combines five key computer vision tasks into a single, cohesive interface:

Image Captioning:

Automatically generates a descriptive sentence about the contents of the image.

Allows for "controllable captioning" by letting the user provide a starting prompt to guide the output.

Visual Question Answering (VQA):

Lets users ask questions about the image in natural language (e.g., "What color is the car?").

The AI analyzes the image to provide a text-based answer.

Object Detection:

Identifies and draws bounding boxes around multiple objects in an image.

Displays a detailed table listing each detected object and its confidence score.

Image Segmentation & Portrait Mode:

Performs semantic segmentation to understand the exact shape and boundaries of objects.

Applies a "Portrait Mode" effect by blurring the background and keeping the main subject (typically a person) in sharp focus.

Includes the ability to remove the blur effect and revert to the original segmentation mask.

Optical Character Recognition (OCR):

Detects and extracts any text present in the image.

Includes image pre-processing (grayscale conversion, contrast enhancement) to improve accuracy on difficult text, such as logos.

Draws bounding boxes around the detected text on the original image.

🚀 How to Run the Application
Follow these steps to get the application running on your local machine.

1. Prerequisites
Python 3.7 or higher

pip (Python package installer)

2. Installation
Open your terminal or command prompt and install all the required libraries by running the following command:

pip install streamlit pandas torch torchvision transformers easyocr numpy Pillow

3. Running the App
Once the installation is complete, navigate to the directory where you saved app.py and run this command:

streamlit run app.py

A new tab should automatically open in your web browser with the Vision AI application running.

Note: The first time you run the app, it will take several minutes to download the pre-trained AI models. This is a one-time process. Subsequent startups will be much faster as the models will be loaded from the cache.

🤖 Models Used
This application relies on the following pre-trained models:

Image Captioning: Salesforce/blip-image-captioning-large

Visual Question Answering: Salesforce/blip-vqa-base

Object Detection: facebook/detr-resnet-50

Image Segmentation: nvidia/segformer-b5-finetuned-ade-640-640

Optical Character Recognition: easyocr library

📂 File Structure
app.py: The main Python script containing all the application logic, model loading, and Streamlit user interface code.

README.md: This is 