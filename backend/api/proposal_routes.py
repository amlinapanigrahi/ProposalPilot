# backend/api/proposal_routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import ProposalEvaluation
from backend.services.document_parser import extract_text_from_pdf
from backend.services.evaluation_pipeline import EvaluationPipeline
import shutil
import os

router = APIRouter(prefix="/api/proposals", tags=["Proposals"])

pipeline = EvaluationPipeline()


@router.post("/upload")
async def upload_and_evaluate_proposal(
    file: UploadFile = File(...),
    budget: float = Form(...),
    db: Session = Depends(get_db)
):
    """
    Receives a proposal file via form data, extracts text content,
    evaluates features using the ML engine, and saves to the database.
    """
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cache file on server disk: {str(e)}")

    try:
        extracted_text = extract_text_from_pdf(file_path)
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Document parsing failure: {str(e)}")

    proposal_title = file.filename.rsplit('.', 1)[0]

    try:
        evaluation_results = pipeline.evaluate_incoming_proposal(
            extracted_text, budget, proposal_title=proposal_title
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"ML Pipeline execution failure: {str(e)}")

    try:
        conf = evaluation_results["evaluation_verdict"]["confidence_percentage"]

        new_proposal = ProposalEvaluation(
            title=proposal_title,
            pdf_filename=file.filename,
            budget=budget,


            final_score=evaluation_results["metrics"]["final_score"],

            lower_confidence=round(max(0.0, conf - 5.0), 2),
            upper_confidence=round(min(100.0, conf + 5.0), 2),

            status="Completed"
        )

        db.add(new_proposal)
        db.commit()
        db.refresh(new_proposal)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database persistent commit failure: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "proposal_id": new_proposal.id,
        "title": new_proposal.title,
        "analysis": evaluation_results
    }