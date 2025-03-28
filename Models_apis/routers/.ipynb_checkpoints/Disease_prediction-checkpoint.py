from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO
import cv2
import os

# Define FastAPI app
router = APIRouter()  # ✅ Define the router

# MODEL_PATH = "/app/best.pt"
# Path to the saved YOLO model
MODEL_PATH = r"D:\USER\Gradation Project\disease\best.pt"
model = YOLO(MODEL_PATH)


# Directory for saving processed images
OUTPUT_DIR = r"D:\USER\Gradation Project\Output_Disease"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Request Model
class ImageRequest(BaseModel):
    image_path: str  # 🔹 Input image path

@router.post("/detect-disease/")
async def detect_disease(request: ImageRequest):
    image_path = request.image_path  # Get image path from request

    # Validate file exists
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Extract the filename from the input image path
    filename = os.path.basename(image_path)  # Example: 'sample.jpg'

    # Generate output path with the same filename
    detected_image_path = os.path.join(OUTPUT_DIR, f"detected_{filename}")

    # Perform prediction
    results = model.predict(source=image_path, imgsz=640, conf=0.5, iou=0.5)

    # Save the processed image
    for result in results:
        plotted_image = result.plot()
        cv2.imwrite(detected_image_path, plotted_image)  # Save output image

    # Return the image file with dynamic filename
    return FileResponse(detected_image_path, media_type="image/jpeg", filename=f"detected_{filename}")

