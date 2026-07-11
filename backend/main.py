from fastapi import FastAPI
from backend.database import engine, Base
import backend.models  
from backend.api.proposal_routes import router as proposal_router
from fastapi.middleware.cors import CORSMiddleware

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI-Driven Proposal Evaluation Suite Backend",
    description="Automated ML/XAI pipeline analytics engine for R&D proposals.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your HTML/JS frontend file to make requests
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(proposal_router)

@app.get("/")
def read_root():
    return {
        "status": "Online",
        "system": "Proposal Evaluation Engine Gateway",
        "version": "1.0.0"
    }

