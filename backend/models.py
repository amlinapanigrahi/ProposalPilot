from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from backend.database import Base
from pydantic import BaseModel
from typing import List, Dict

class ProposalEvaluation(Base):
    __tablename__ = "proposal_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True, nullable=True)
    pdf_filename = Column(String, nullable=False)
    budget = Column(Float, nullable=True)
    
    final_score = Column(Float, nullable=True)       # Range: 0.0 - 100.0
    lower_confidence = Column(Float, nullable=True)  # Uncertainty lower bound
    upper_confidence = Column(Float, nullable=True)  # Uncertainty upper bound
    
    report_pdf_path = Column(String, nullable=True)
    status = Column(String, default="Pending")        # Pending, Processing, Completed, Failed

    created_at = Column(DateTime, default=datetime.utcnow)

class MetricScores(BaseModel):
    novelty_score: float
    novelty_rating: str
    financial_score: float
    financial_rating: str

class XAIAttribution(BaseModel):
    novelty_impact_weight: float
    financial_feasibility_weight: float
    technical_density_weight: float
    methodology_type: str

class EvaluationVerdict(BaseModel):
    decision: str
    confidence_percentage: float

class AnalysisPayload(BaseModel):
    status: str
    metrics: MetricScores
    explainable_ai_attribution: XAIAttribution
    ai_generated_narrative: str
    similar_past_projects: List[Dict]
    evaluation_verdict: EvaluationVerdict

class ProposalResponse(BaseModel):
    proposal_id: int
    title: str
    analysis: AnalysisPayload