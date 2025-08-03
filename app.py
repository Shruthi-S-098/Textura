import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import io

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Textura – Handwritten OCR",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- HIDE STREAMLIT BRANDING ----
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .st-emotion-cache-18ni7ap {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- APP TITLE ----
st.title("🖋️ Textura – Handwritten Text Recognizer")
st.write("Upload a handwritten image (JPG or PNG) to extract text using Google Vision OCR.")

# ---- GOOGLE CLOUD VISION SETUP ----
creds_dict = st.secrets["google"]
credentials = service_account.Credentials.from_service_account_info(dict(creds_dict))
client = vision.ImageAnnotatorClient(credentials=credentials)

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # ---- OCR PROCESSING ----
    with st.spinner("Extracting text with Google Vision..."):
        # Convert image to bytes
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        content = image_bytes.getvalue()

        # Google Vision API
        vision_image = vision.Image(content=content)
        response = client.document_text_detection(
            image=vision_image,
            image_context={"language_hints": ["en-t-i0-handwrit"]}
        )

    # ---- TEXT OUTPUT ----
    extracted_text = response.full_text_annotation.text.strip()
    if extracted_text:
        st.subheader("🧾 Extracted Text")
        st.text_area("Result", extracted_text, height=300)
    else:
        st.warning("No readable text was extracted.")

    # ---- WORD CONFIDENCE ----
    st.subheader("📊 Word-wise Confidence")
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = ''.join([s.text for s in word.symbols])
                    st.write(f"🔹 {word_text} — confidence: {word.confidence:.2f}")
