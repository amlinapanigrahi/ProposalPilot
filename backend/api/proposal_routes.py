# backend/api/proposal_routes.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import ProposalEvaluation
from backend.services.document_parser import DocumentParser

router = APIRouter(prefix="/api/proposals", tags=["Proposals"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_proposal(
    file: UploadFile = File(...),
    budget: float = Form(None),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid format. Only PDF files are allowed.")
        
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        raw_text = DocumentParser.extract_text_from_pdf(file_path)
        parsed_sections = DocumentParser.parse_sections(raw_text)
        
        new_evaluation = ProposalEvaluation(
            title=file.filename.replace(".pdf", ""),
            pdf_filename=file.filename,
            budget=budget,
            status="Processing"
        )
        
        db.add(new_evaluation)
        db.commit()
        db.refresh(new_evaluation)
        
        return {
            "message": "Proposal uploaded and text parsed successfully.",
            "proposal_id": new_evaluation.id,
            "detected_sections": {
                "abstract_length": len(parsed_sections["abstract"]),
                "methodology_length": len(parsed_sections["methodology"]),
                "budget_segment_length": len(parsed_sections["budget_details"])
            }
        }
        
    except Exception as e:
        # Cleanup file if writing data metrics crashes
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Parsing Engine failure: {str(e)}")