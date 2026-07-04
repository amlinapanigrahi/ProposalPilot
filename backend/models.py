from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime
from backend.database import Base

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