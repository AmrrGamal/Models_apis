from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import tempfile
import os

router = APIRouter()

# ===== NEW MODEL LOADING SYSTEM =====
MODEL_DIR = os.path.join(os.getcwd(), "models")

def get_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(500, detail=f"Model {model_name} not found")
    return YOLO(model_path)

model = get_model("best.pt")  # Using the new loader
# ===== END OF NEW SYSTEM =====

@router.post("")
async def detect_disease(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(400, detail="Only JPG/PNG images allowed")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            content = await file.read()
            if not content:
                raise HTTPException(400, detail="Empty file")
            tmp_file.write(content)
            input_path = tmp_file.name

        results = model.predict(source=input_path, imgsz=640, conf=0.5)
        
        output_path = tempfile.mktemp(suffix=".jpg")
        for result in results:
            cv2.imwrite(output_path, result.plot())

        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename=f"processed_{file.filename}"
        )

    except Exception as e:
        raise HTTPException(500, detail=str(e))
    finally:
        # Clean up temp files
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass