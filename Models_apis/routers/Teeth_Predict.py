from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import tempfile
import os

router = APIRouter()

# ===== ADD THIS SECTION =====
MODEL_DIR = os.path.join(os.getcwd(), "models")

def get_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        raise HTTPException(500, detail=f"Model {model_name} not found")
    return YOLO(model_path)

model = get_model("best_count.pt")  # Changed from direct MODEL_PATH
# ===== END OF ADDED SECTION =====

molars = {7, 6, 5, 13, 14, 15, 31, 30, 29, 21, 22, 23}
incisors = {0, 1, 8, 9, 24, 25, 16, 17}
canines = {2, 10, 18, 26}
premolars = {3, 4, 11, 12, 19, 20, 27, 28}

@router.post("")
async def detect_teeth(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, detail="Only image files allowed")

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            content = await file.read()
            if not content:
                raise HTTPException(400, detail="Empty file")
            tmp_file.write(content)
            tmp_file.flush() 
            
            results = model.predict(source=tmp_file.name, imgsz=640, conf=0.5)

        detected_teeth = set()
        for result in results:
            for box in result.boxes:
                detected_teeth.add(int(box.cls))

        return {
            "total_teeth": min(len(detected_teeth), 32),
            "incisors": len(detected_teeth & incisors),
            "canines": len(detected_teeth & canines),
            "premolars": len(detected_teeth & premolars),
            "molars": len(detected_teeth & molars),
            "missing_teeth": max(32 - len(detected_teeth), 0),
            "message": "No missing teeth" if len(detected_teeth) == 32 
                      else f"Missing {32 - len(detected_teeth)} teeth"
        }

    except Exception as e:
        raise HTTPException(500, detail=str(e))