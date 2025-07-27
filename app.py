import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
)
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import easyocr
import numpy as np
import pandas as pd


# Set page config
st.set_page_config(layout="wide", page_title="Vision AI")


# Caching the models for faster loading
@st.cache_resource
def load_captioning_model():
    """Load the captioning model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )
    return {"caption_proc": processor, "caption_model": model}


@st.cache_resource
def load_vqa_model():
    """Load the VQA model and processor."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return {"vqa_proc": processor, "vqa_model": model}


@st.cache_resource
def load_object_detection_model():
    """Load the object detection model and processor."""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    return {"obj_proc": processor, "obj_model": model}


@st.cache_resource
def load_segmentation_model():
    """Load the segmentation model and processor."""
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-ade-640-640"
    )
    return {"seg_proc": processor, "seg_model": model}


@st.cache_resource
def load_ocr_reader():
    """Load the OCR reader model."""
    return {"ocr_reader": easyocr.Reader(["en"])}


# Models


# 1. Captioning Model
def perform_captioning(image, prompt, models):
    """Generate a caption for the image, with an optional prompt."""
    if prompt:
        # Conditional captioning
        inputs = models["caption_proc"](image, text=prompt, return_tensors="pt")
    else:
        # Unconditional captioning
        inputs = models["caption_proc"](image, return_tensors="pt")
    out = models["caption_model"].generate(**inputs, max_length=50)
    return models["caption_proc"].decode(out[0], skip_special_tokens=True)


# 2. VQA Model
def perform_vqa(image, question, models):
    """Answer a question about the image."""
    if not question:
        return "Please ask a question!"
    inputs = models["vqa_proc"](image, question, return_tensors="pt")
    out = models["vqa_model"].generate(**inputs)
    return models["vqa_proc"].decode(out[0], skip_special_tokens=True)


# 3. Object Detection Model
def perform_object_detection(image, models):
    """Detect objects, draw boxes, and return the results as a table."""
    inputs = models["obj_proc"](images=image, return_tensors="pt")
    outputs = models["obj_model"](**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = models["obj_proc"].post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    draw = ImageDraw.Draw(image)
    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        label_text = models["obj_model"].config.id2label[label.item()]
        # Add data to the list for the table
        detections.append(
            {
                "Object": label_text.capitalize(),
                "Confidence": f"{score.item()*100:.2f}%",
            }
        )
        # Draw the bounding box and label on the image
        draw.rectangle(box, outline="red", width=3)
        draw.text(
            (box[0], box[1]), f"{label_text} ({round(score.item(), 2)})", fill="red"
        )

    # Create a DataFrame from the detections
    df = pd.DataFrame(detections)
    return image, df


# 4. Image Segmentation Model
def perform_segmentation(image, models):
    """
    Performs segmentation and returns both the visual mask image and the raw mask data.
    """
    inputs = models["seg_proc"](images=image, return_tensors="pt")
    outputs = models["seg_model"](**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    # Create a color palette for visualization
    ade_palette = np.random.randint(
        0, 255, (models["seg_model"].config.num_labels, 3), dtype=np.uint8
    )
    color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
    for label in torch.unique(pred_seg):
        color_seg[pred_seg == label, :] = ade_palette[label]

    # Create the visual overlay image
    mask_image = Image.fromarray(color_seg).convert("RGB")
    visual_result = Image.blend(image, mask_image, alpha=0.5)
    return visual_result, pred_seg  # Return both the visual and the raw data


# 4. Image Segmentation Model - Blur Effect
def apply_portrait_from_mask(original_image, pred_seg):
    """
    Applies a portrait blur using a pre-computed segmentation mask.
    """
    # Create a blurred version of the original image
    blurred_image = original_image.filter(ImageFilter.GaussianBlur(15))

    # Create a mask from the raw data to isolate the person (label 12)
    foreground_mask = (pred_seg == 12).numpy().astype("uint8") * 255
    foreground_mask = Image.fromarray(foreground_mask).convert("L")

    # Composite the final image
    final_image = Image.composite(original_image, blurred_image, foreground_mask)

    return final_image


# 5. Optical Character Recognition (OCR) Model
def perform_ocr(image, models):
    """Perform OCR on the image, with pre-processing to improve accuracy."""
    # 1. Convert the image to grayscale to improve contrast
    processed_image = image.convert("L")
    # 2. Enhance the contrast of the grayscale image
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(2.0)  # The factor 2.0 is a good starting point

    image_np = np.array(processed_image)
    results = models["ocr_reader"].readtext(image_np)
    draw = ImageDraw.Draw(image)  # Draw on the original color image
    full_text = []

    if results:
        for bbox, text, prob in results:
            full_text.append(text)
            # Draw bounding box on the original image
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            draw.rectangle([top_left, bottom_right], outline="green", width=2)
    return image, "\n".join(full_text)


# Main function to run the Streamlit app
def main():
    st.markdown(
        "<h1 style='font-size: 36px;'>🧿 Vision AI: Beyond the Image, Into the Insight</h1>",
        unsafe_allow_html=True,
    )

    # Sidebar for image upload and task selection
    st.sidebar.header("1. Upload Your Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"]
    )

    # Initialize session state to remember results across reruns
    if "results" not in st.session_state:
        st.session_state.results = {}

    # Logic to clear results when a new image is uploaded
    if uploaded_file is not None:
        if (
            "last_uploaded_file_id" not in st.session_state
            or st.session_state.last_uploaded_file_id != uploaded_file.file_id
        ):
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            st.session_state.results = {}  # Clear previous results

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.sidebar.header("2. Choose an AI Task")
        task = st.sidebar.selectbox(
            "Select a task:",
            [
                "Image Captioning",
                "Visual Question Answering",
                "Object Detection",
                "Image Segmentation",
                "Optical Character Recognition (OCR)",
            ],
        )

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("AI is thinking..."):

            # Process the selected task

            if task == "Image Captioning":
                models = load_captioning_model()
                # This block runs automatically only once, then waits for user input
                if "caption" not in st.session_state.results:
                    caption = perform_captioning(image, "", models)
                    st.session_state.results["caption"] = caption
                # Add a prompt and button for refining the caption
                prompt = st.sidebar.text_input(
                    "Refine caption with a prompt:", key="caption_prompt"
                )
                # If the button is clicked, run the model and SAVE the result
                if st.sidebar.button("Regenerate Caption", use_container_width=True):
                    with st.spinner("Rethinking the caption..."):
                        caption = perform_captioning(image, prompt, models)
                        st.session_state.results["caption"] = caption
                # Always display the caption that is currently saved in the state
                col2.subheader("Generated Caption:")
                col2.write(
                    f"> *{st.session_state.results.get('caption', 'Processing...')}*"
                )

            elif task == "Visual Question Answering":
                models = load_vqa_model()
                # Add this informational message to set user expectations
                st.info(
                    "The AI will attempt to answer any question, but it may be inaccurate if the answer isn't clear from the image."
                )
                question = st.sidebar.text_input(
                    "Ask a question about the image:", key="vqa_question"
                )
                if st.sidebar.button("Get Answer", use_container_width=True):
                    # When button is clicked, run the model and SAVE the result
                    answer = perform_vqa(image, question, models)
                    st.session_state.results["vqa"] = {
                        "question": question,
                        "answer": answer,
                    }
                # Always check the state and DISPLAY the saved result
                if "vqa" in st.session_state.results:
                    vqa_results = st.session_state.results["vqa"]
                    col2.subheader(f"Question: ")
                    col2.write(f"> {vqa_results['question']}")
                    col2.subheader(f"Answer: ")
                    col2.write(f"> **{vqa_results['answer']}**")

                # Small Note
                st.caption(
                    "Note: Use 'Extract Text (OCR)' for questions about text (like names or dates) and 'Visual Question Answering' for questions about the scene (like colors or actions)."
                )

            elif task == "Object Detection":
                models = load_object_detection_model()
                if st.sidebar.button("Detect Objects", use_container_width=True):
                    # When button is clicked, run the model and SAVE the result
                    result_image, detection_df = perform_object_detection(
                        image.copy(), models
                    )
                    st.session_state.results["obj_detection"] = {
                        "image": result_image,
                        "df": detection_df,
                    }
                if "obj_detection" in st.session_state.results:
                    obj_results = st.session_state.results["obj_detection"]
                    col2.image(
                        obj_results["image"],
                        caption="Processed Image with Objects Detected",
                        use_container_width=True,
                    )
                    st.subheader("Detected Objects")
                    if not obj_results["df"].empty:
                        st.dataframe(obj_results["df"], use_container_width=True)
                    else:
                        st.info("No objects detected with high confidence.")

            elif task == "Image Segmentation":
                models = load_segmentation_model()
                # Automatically perform segmentation the first time this task is viewed for an image
                if "segmentation" not in st.session_state.results:
                    visual_image, raw_mask = perform_segmentation(image.copy(), models)
                    # Save the initial visual, the current visual, and the raw mask data
                    st.session_state.results["segmentation"] = {
                        "initial_visual": visual_image,
                        "current_visual": visual_image,
                        "mask": raw_mask,
                    }
                # Display the current visual result
                current_visual = st.session_state.results["segmentation"][
                    "current_visual"
                ]
                col2.image(
                    current_visual, caption="Processed Image", use_container_width=True
                )

                # Create two columns for the side-by-side buttons
                btn_col1, btn_col2 = st.columns(2)

                # Button to apply the blur effect
                if btn_col1.button("Apply Blur Effect", use_container_width=True):
                    with st.spinner("Applying blur..."):
                        original_image = image.copy()
                        raw_mask = st.session_state.results["segmentation"]["mask"]
                        portrait_image = apply_portrait_from_mask(
                            original_image, raw_mask
                        )
                        # Overwrite the current visual with the new portrait image
                        st.session_state.results["segmentation"][
                            "current_visual"
                        ] = portrait_image
                        st.rerun()

                # New button to remove the blur effect
                if btn_col2.button("Remove Blur Effect", use_container_width=True):
                    # Revert the current visual back to the initial segmentation mask
                    st.session_state.results["segmentation"]["current_visual"] = (
                        st.session_state.results["segmentation"]["initial_visual"]
                    )
                    st.rerun()

                # Add the small comment under the buttons
                st.caption(
                    "Note: The portrait blur effect works best when a person is the main subject of the image."
                )

            elif task == "Optical Character Recognition (OCR)":
                models = load_ocr_reader()
                st.caption(
                    "Note: OCR works best on clear text. Logos or stylized fonts can sometimes be detected incorrectly."
                )

                if st.sidebar.button("Extract Text", use_container_width=True):
                    result_image, extracted_text = perform_ocr(image.copy(), models)
                    # When button is clicked, SAVE the result to the state
                    st.session_state.results["ocr"] = {
                        "image": result_image,
                        "text": extracted_text,
                        "original_image": image,
                    }
                # Always check the state and DISPLAY the saved result
                if "ocr" in st.session_state.results:
                    ocr_results = st.session_state.results["ocr"]
                    if ocr_results["text"].strip():
                        # This line ensures the processed image fits the column
                        col2.image(
                            ocr_results["image"],
                            caption="Processed Image with Text Detected",
                            use_container_width=True,
                        )
                        st.subheader("Extracted Text:")
                        st.markdown(f"```\n{ocr_results['text']}\n```")
                    else:
                        # This line ensures the original image fits when no text is found
                        col2.image(
                            image,
                            caption="Original Image (No Text Detected)",
                            use_container_width=True,
                        )
                        st.info("No text was detected in the image.")

    else:
        st.info("Please upload an image using the sidebar to get started.")


if __name__ == "__main__":
    main()
