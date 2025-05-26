import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from typing import Dict, List
from dataclasses import dataclass
from rapidfuzz import fuzz, process
import tkinter as tk
from tkinter import simpledialog
import difflib 

def get_reference_name_gui(ocr_text: str = "") -> str:
    root = tk.Tk()
    root.withdraw()  # Hide main window
    name = simpledialog.askstring("Input", "Enter the reference full name:")

    if not name:
        return ""

    if ocr_text:
        similarity = fuzz.token_set_ratio(name.upper(), ocr_text.upper())
        print(f"📊 Input vs OCR Similarity: {similarity}%")
        if similarity < 50:
            print("⚠️  Warning: Low similarity between input and OCR result.")

    return name

def find_best_line_match(input_name, ocr_text):
    best_match = None
    best_score = 0.0

    for line in ocr_text:
        line_clean = line.strip()
        score = difflib.SequenceMatcher(None, input_name.lower(), line_clean.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = line_clean

    return best_match, best_score


# =================== CONFIGURATION ===================

LICENSE_FIELDS = [
    "REPUBLIC OF THE PHILIPPINES", "DEPARTMENT OF TRANSPORTATION", "LAND TRANSPORTATION OFFICE",
    "DRIVER'S LICENSE", "Last Name", "First Name", "Middle Name", "Nationality", "Sex",
    "Date of Birth", "Weight", "Height", "Address", "License No.", "Expiration Date",
    "Agency Code", "Blood Type", "Eyes Color", "DL Codes", "Conditions"
]

@dataclass
class LicenseInfo:
    document_type: str
    name: str
    document_verified: str
    formatted_text: str

@dataclass
class NameInfo:
    document_type: str
    name: str
    document_verified: str
    formatted_text: str

def package_name_info(structured_data: Dict[str, str], basic_text: str) -> NameInfo:
    return NameInfo(
        document_type="Driver's License",
        name=structured_data.get('Name', 'Not Found'),
        document_verified=structured_data.get('Document Verified', 'Unverified'),
        formatted_text=format_text_output(basic_text)
    )


# =================== ENHANCED IMAGE PREPROCESSING ===================

def preprocess_image(image_path: str) -> np.ndarray:
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path}")
    
    # Apply basic preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    bilateral = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # Apply Otsu's thresholding
    thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Apply morphological operations to clean the image
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply denoising as the final step
    denoised = cv2.fastNlMeansDenoising(opening, None, 10, 7, 21)
    
    return denoised

def enhance_image(image: np.ndarray) -> np.ndarray:
    # Sharpen the image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Improve contrast and brightness
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.5, beta=10)
    
    return enhanced

def preprocess_batch(image_path: str) -> List[np.ndarray]:
    """Generate multiple preprocessed versions of the image for OCR attempts"""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path}")
    
    processed_images = []
    
    # Version 1: Standard preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(thresh)
    
    # Version 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    thresh2 = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(thresh2)
    
    # Version 3: Bilateral filtering with adaptive thresholding
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    adaptive_thresh = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    processed_images.append(adaptive_thresh)
    
    # Version 4: Rescaled and denoised
    img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray_resized, None, 10, 7, 21)
    thresh_resized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    processed_images.append(thresh_resized)
    
    # Version 5: Edge enhancement
    edges = cv2.Canny(gray, 100, 200)
    dilated = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
    processed_images.append(255 - dilated)  # Invert for better text recognition
    
    return processed_images

def extract_text_from_image(image_path: str, config: str = '--psm 11 --oem 3') -> str:
    try:
        img = preprocess_image(image_path)
        enhanced = enhance_image(img)
        return pytesseract.image_to_string(enhanced, config=config)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_text_with_multiple_methods(image_path: str) -> List[str]:
    results = []
    processed_images = preprocess_batch(image_path)
    
    for i, img in enumerate(processed_images):
        for psm in [3, 6, 11]:
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(img, config=custom_config)
            results.append(text)
    
    return results


def extract_structured_data(image_path: str) -> Dict[str, str]:
    img = enhance_image(preprocess_image(image_path))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    full_text = ' '.join([t for t in data['text'] if t.strip()])
    extracted_data = {}

    for field in LICENSE_FIELDS:
        match = re.search(rf"{field}[:\-]?\s*(.*)", full_text, re.IGNORECASE)
        if match:
            extracted_data[field] = match.group(1).strip()[:50]  # Limit long field text

    return extracted_data

def extract_name_from_lines(image_path: str, reference_name: str = "", best_ocr_match: str = "", match_score: float = 0.0) -> Dict[str, str]:
    """Simplified name extraction that prioritizes the best match found"""
    
    # Check header keywords for verification
    preprocessed_images = preprocess_batch(image_path)
    best_text = ""
    max_length = 0

    for img in preprocessed_images:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        if len(text) > max_length:
            best_text = text
            max_length = len(text)

    raw_text = best_text if max_length >= 50 else pytesseract.image_to_string(cv2.imread(image_path))
    full_text = " ".join(raw_text.splitlines()).upper()

    # Combine both LICENSE_FIELDS and HEADER_KEYWORDS for verification
    VERIFICATION_KEYWORDS = [
        "REPUBLIC", "PHILIPPINES", "DEPARTMENT", "TRANSPORTATION", 
        "LAND TRANSPORTATION OFFICE", "DRIVER'S LICENSE", "DRIVERS LICENSE",
        "LICENSE", "LAST NAME", "FIRST NAME", "MIDDLE NAME", "NATIONALITY", 
        "DATE OF BIRTH", "ADDRESS", "LICENSE NO", "EXPIRATION DATE"
    ]
    
    matched_keywords = {kw for kw in VERIFICATION_KEYWORDS if kw in full_text}
    is_verified = len(matched_keywords) >= 2  # Only need 2+ matches for verification

    name_info = {}
    
    # If we have a good match (60%+) and reference name, use the reference name
    if reference_name and match_score >= 0.6:
        name_info["Name"] = reference_name
        name_info["Matched From"] = "User Input (High Confidence Match)"
        name_info["Match Confidence"] = f"{match_score * 100:.1f}%"
        name_info["Document Verified"] = "Driver's License Detected" if is_verified else "Unverified Document"
        return name_info
    
    # Otherwise use the best OCR match if available
    if best_ocr_match and match_score > 0.4:  # Lower threshold for OCR match
        name_info["Name"] = best_ocr_match
        name_info["Matched From"] = "Best OCR Line Match"
        name_info["Match Confidence"] = f"{match_score * 100:.1f}%"
        name_info["Document Verified"] = "Driver's License Detected" if is_verified else "Unverified Document"
        return name_info
    
    # Fallback: try to find any reasonable name in the text
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    for line in lines:
        clean = re.sub(r"[^A-Z\s,.]", "", line.upper()).strip()
        
        # Skip header lines
        if any(header in clean for header in VERIFICATION_KEYWORDS):
            continue
            
        # Look for name-like patterns
        if 4 < len(clean) < 60 and clean.replace(" ", "").isalpha() and " " in clean:
            name_info["Name"] = clean
            name_info["Matched From"] = "Pattern Detection"
            name_info["Document Verified"] = "Driver's License Detected" if is_verified else "Unverified Document"
            return name_info
    
    # No name found
    name_info["Name"] = "Not Found"
    name_info["Document Verified"] = "Driver's License Detected" if is_verified else "Unverified Document"
    return name_info

def format_text_output(raw_text: str) -> str:
    lines = raw_text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        sanitized = re.sub(r"[^a-zA-Z0-9\s,\.]", "", line)
        if len(sanitized) >= 3 and any(c.isalpha() for c in sanitized):
            cleaned.append(sanitized)
    return "\n".join(cleaned)


def package_license_info(structured_data: Dict[str, str], basic_text: str) -> LicenseInfo:
    return LicenseInfo(
        document_type=structured_data.get('Document Type', 'Unknown'),
        name=structured_data.get('Name', 'Not Found'),
        document_verified=structured_data.get('Document Verified', 'Unverified'),
        formatted_text=format_text_output(basic_text)
    )

def draw_license_box(image_path: str, output_path: str = "boxed_license.jpg") -> str:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 300 and h > 150:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                break  # Only draw the largest valid one

    cv2.imwrite(output_path, img)
    return output_path

# =================== MAIN EXECUTION ===================

def licenseRead(image_path: str, reference_name: str = "", best_ocr_match: str = "", match_score: float = 0.0):
    print(f"✅ Image saved to {image_path}\n")

    boxed_image_path = draw_license_box(image_path)
    print(f"🖼️  Bounding Box Image saved as: {boxed_image_path}")

    basic_text = extract_text_from_image(image_path)
    structured_data = extract_name_from_lines(image_path, reference_name=reference_name, 
                                            best_ocr_match=best_ocr_match, match_score=match_score)

    packaged = package_name_info(structured_data, basic_text)

    # 🖨️ Improved, user-friendly log
    print("🧾 ===== OCR SUMMARY =====")
    print(f"🆔 Document Type     : {packaged.document_type}")
    print(f"🧠 Detected Name     : {packaged.name}")
    print(f"🔐 Verification      : {packaged.document_verified}")
    if "Match Confidence" in structured_data:
        print(f"🎯 Match Confidence  : {structured_data['Match Confidence']}")
        print(f"📥 Name Matched From  : {structured_data['Matched From']}")
    print("\n🧾 Extracted Text:")
    print(packaged.formatted_text)
    print("=========================\n")

    return packaged


def auto_capture_license(output_path="license.jpg", reference_name=""):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return None

    print("📸 Auto-capture mode: Position license in the box. Capturing when driver's license is detected...")
    if reference_name:
        print(f"🎯 Looking for license matching: {reference_name}")
    print("Press 'q' to quit manual mode.")
    
    frame_count = 0
    detection_threshold = 3  # Need 3 consecutive detections to capture
    consecutive_detections = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Get image dimensions
        h, w = frame.shape[:2]

        # Define fixed box size and position (centered)
        box_width, box_height = 500, 300
        top_left = (w // 2 - box_width // 2, h // 2 - box_height // 2)
        bottom_right = (w // 2 + box_width // 2, h // 2 + box_height // 2)

        # Crop the region of interest (inside the box)
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Every 10 frames, check for license detection
        frame_count += 1
        if frame_count % 10 == 0:  # Check every 10 frames to avoid too much processing
            try:
                # Quick OCR on the ROI
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                thresh_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                quick_text = pytesseract.image_to_string(thresh_roi, config='--psm 6 --oem 3').upper()
                
                # Check for license keywords
                VERIFICATION_KEYWORDS = [
                    "REPUBLIC", "PHILIPPINES", "DEPARTMENT", "TRANSPORTATION", 
                    "LAND TRANSPORTATION OFFICE", "DRIVER'S LICENSE", "DRIVERS LICENSE",
                    "LICENSE", "LAST NAME", "FIRST NAME", "MIDDLE NAME", "NATIONALITY"
                ]
                
                matched_keywords = sum(1 for kw in VERIFICATION_KEYWORDS if kw in quick_text)
                
                if matched_keywords >= 2:
                    consecutive_detections += 1
                    print(f"🔍 License detected! ({consecutive_detections}/{detection_threshold})")
                else:
                    consecutive_detections = 0
                    
            except Exception as e:
                consecutive_detections = 0

        # Draw the guide box (green if detecting, red if not)
        box_color = (0, 255, 0) if consecutive_detections > 0 else (0, 0, 255)
        cv2.rectangle(frame, top_left, bottom_right, box_color, 2)
        
        # Add status text
        status_text = f"Detecting... {consecutive_detections}/{detection_threshold}" if consecutive_detections > 0 else "Position license in box"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Add reference name to display if provided
        if reference_name:
            cv2.putText(frame, f"Target: {reference_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Auto License Capture", frame)

        # Auto capture when threshold is reached
        if consecutive_detections >= detection_threshold:
            cv2.imwrite(output_path, frame)
            print(f"✅ License auto-captured and saved to {output_path}")
            break

        # Allow manual quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("❌ Auto-capture canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return output_path if consecutive_detections >= detection_threshold else None


if __name__ == "__main__":
    # ✅ Get the reference name FIRST before capturing
    print("👋 Welcome to License Reader!")
    print("Please enter the full name to match against the license...")
    
    reference_input = get_reference_name_gui("")
    
    if not reference_input:
        print("❌ No name provided. Exiting...")
        exit()
    
    print(f"🎯 Target name: {reference_input}")
    print("Now let's capture the license card...")
    
    # Capture the license with the reference name for display
    image_path = auto_capture_license(reference_name=reference_input)
    
    if image_path:
        print("🔍 Processing captured license...")
        
        # Extract OCR preview for matching
        ocr_preview = extract_text_from_image(image_path)
        ocr_lines = [line.strip() for line in ocr_preview.splitlines() if line.strip()]

        # Find the line with highest similarity to the user input
        name_from_ocr, sim_score = find_best_line_match(reference_input, ocr_lines)

        print(f"📊 Best OCR Line Match: '{name_from_ocr}' with {sim_score * 100:.2f}% similarity")

        # ✅ Process the license with all the similarity detection intact
        result = licenseRead(image_path, reference_name=reference_input, 
                           best_ocr_match=name_from_ocr, match_score=sim_score)

        print(result)
    else:
        print("❌ No license captured. Exiting...")