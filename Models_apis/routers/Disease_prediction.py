from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import tempfile
import os

router = APIRouter()

MODEL_PATH = os.path.join(os.getcwd(), "models", "best.pt")
model = YOLO(MODEL_PATH)

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