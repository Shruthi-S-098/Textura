import os
from google.cloud import vision
import io

# STEP 1: Set this only if not already done in CMD/PowerShell
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\\task-3\\cool-component-465106-h0-560ee7e185c0.json"
# STEP 2: Initialize Vision client
client = vision.ImageAnnotatorClient()

# STEP 3: Load the image (update with your file path)
image_path = r"D:\task-3\vision-image.jpg"  # <- CHANGE THIS

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# STEP 4: Detect handwriting
response = client.document_text_detection(
    image=image,
    image_context={"language_hints": ["en-t-i0-handwrit"]}  # this helps for cursive/handwriting
)

# STEP 5: Print the recognized text
print("\n=== RECOGNIZED TEXT ===\n")
print(response.full_text_annotation.text)

# Optional: Loop through each word with confidence
for page in response.full_text_annotation.pages:
    for block in page.blocks:
        for paragraph in block.paragraphs:
            for word in paragraph.words:
                word_text = ''.join([symbol.text for symbol in word.symbols])
                print(f"{word_text} (confidence: {word.confidence:.2f})")
