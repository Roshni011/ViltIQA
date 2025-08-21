```python
import streamlit as st
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

# Configure Streamlit page
st.set_page_config(page_title="VQA with ViLT", layout="wide")

# Load ViLT processor and model
@st.cache_resource
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Function to get answer from model
def get_answer(image: Image.Image, question: str):
    try:
        encoding = processor(image, question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]
            confidence = probs[0, idx].item()
        return answer, confidence
    except Exception as e:
        return str(e), 0.0

# Streamlit UI
st.title("🖼️ Visual Question Answering (ViLT)")
st.write("Upload an image and ask a question about it.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    question = st.text_input("Enter your question")
    if image is not None and question:
        if st.button("Ask Question"):
            answer, confidence = get_answer(image, question)
            st.success(f"Answer: {answer}")
            st.caption(f"Confidence: {confidence:.2f}")
            image = Image.open(uploaded_file)
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='JPEG')
            image_bytes = image_byte_array.getvalue()

            # Get the answer
            answer = get_answer(image_bytes, question)

            # Display the answer
            st.success("Answer: " + answer)

