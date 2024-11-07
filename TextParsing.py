import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_path
import os
import re
import pandas as pd
from nltk.corpus import stopwords

# Load YOLOv8 model
model = YOLO("Training010624/weights/best.pt")  

# Initialize EasyOCR
reader = easyocr.Reader(['en'])  

# Set of stop words from NLTK
STOP_WORDS = set(stopwords.words('english'))

def convert_pdf_to_images(pdf_path):
    """
    Converts a PDF file into a list of images (one per page).
    """
    images = convert_from_path(pdf_path)
    return [np.array(img) for img in images]

def detect_sections(image):
    """
    Uses YOLOv8 to detect sections in a CV image.
    Returns bounding boxes for each detected section.
    """
    results = model(image)
    sections = []
    
    for result in results[0].boxes:  # Access the detected boxes
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        label = result.cls  # Detected section label
        confidence = result.conf[0]  # Detection confidence
        sections.append({'box': (x1, y1, x2, y2), 'label': label, 'confidence': confidence})
        
    return sections

def extract_text_from_section(image, box):
    """
    Uses EasyOCR to extract text from a specific region in an image.
    """
    x1, y1, x2, y2 = box
    section_img = image[y1:y2, x1:x2]
    text = reader.readtext(section_img, detail=0)
    return " ".join(text)

def process_cv(input_path):
    """
    Processes a CV, detects sections, and extracts text from each section.
    """
    # Convert PDF to images if the file is a PDF
    if input_path.lower().endswith('.pdf'):
        images = convert_pdf_to_images(input_path)
    else:
        images = [cv2.imread(input_path)]

    all_sections_text = []
    
    for page_num, image in enumerate(images):
        # Detect sections
        sections = detect_sections(image)
        
        for section in sections:
            # Extract text from each section
            box = section['box']
            label = section['label']
            text = extract_text_from_section(image, box)
            all_sections_text.append({'page': page_num + 1, 'label': label, 'text': text})
            
            # Display for verification
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} ({section['confidence']:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save annotated images for verification
        output_image_path = f"output_page_{page_num + 1}.jpg"
        cv2.imwrite(output_image_path, image)
    
    return all_sections_text


def save_to_csv(sections_text, csv_path='data.csv'):
    """
    Save CV information to a CSV file, with separate entries for each section.
    """
    # Clean each section's text and prepare data for the DataFrame
    data = [
        {'text': section['text']}
        for section in sections_text
    ]
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

# Example usage
input_path = 'CV_TEST.jpg'  
sections_text = process_cv(input_path)

# Save cleaned text from all sections to CSV
save_to_csv(sections_text)

# Display extracted and cleaned text for each section
for section in sections_text:
    print(f"Text: {section['text']}\n")
