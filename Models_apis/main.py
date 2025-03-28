from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import Teeth_Predict, Disease_prediction

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(Disease_prediction.router, prefix="/detect-disease", tags=["Diseases"])
app.include_router(Teeth_Predict.router, prefix="/detect-teeth", tags=["Teeth"])

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 

