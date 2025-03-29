from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import Teeth_Predict, Disease_prediction
import os  # NEW IMPORT

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(Disease_prediction.router, prefix="/detect-disease", tags=["Diseases"])
app.include_router(Teeth_Predict.router, prefix="/detect-teeth", tags=["Teeth"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",  # MUST be 0.0.0.0
        port=int(os.environ.get("PORT", 8080))
    )