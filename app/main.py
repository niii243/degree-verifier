# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import easyocr
import cv2
import os
import tempfile
import shutil

app = FastAPI(title="Degree Verifier", version="0.3")

# Initialize EasyOCR reader once (expensive to reload every request)
reader = easyocr.Reader(['en', 'hi'], gpu=False)

def preprocess_image(image_path: str) -> str:
    """
    Preprocess image using OpenCV:
    - Convert to grayscale
    - Apply binary threshold
    - Save processed image
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return image_path  # fallback if not a valid image

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold (binarization)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Save processed image to a temporary path
    processed_fd, processed_path = tempfile.mkstemp(suffix=".jpg")
    os.close(processed_fd)  # close file descriptor
    cv2.imwrite(processed_path, thresh)

    return processed_path

@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Degree Verifier API running"}

@app.post("/upload")
async def upload_and_ocr(file: UploadFile = File(...)):
    """
    Handles image upload:
    - Saves to a temporary file
    - Preprocesses with OpenCV
    - Extracts text with EasyOCR
    - Deletes temp files after processing
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Ensure it's an image
    ext = file.filename.lower().split(".")[-1]
    if ext not in ["jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported")

    # Create a temporary file for the uploaded image
    suffix = "." + ext
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(tmp_fd)  # close file descriptor

    with open(tmp_path, "wb") as tmp:
        shutil.copyfileobj(file.file, tmp)

    try:
        # Preprocess image with OpenCV
        processed_path = preprocess_image(tmp_path)

        # Run OCR
        results = reader.readtext(processed_path, detail=0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    finally:
        # Cleanup temp files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)

    return {
        "filename": file.filename,
        "ocr_text": results
    }