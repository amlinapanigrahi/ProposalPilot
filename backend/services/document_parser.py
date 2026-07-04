import os
from pypdf import PdfReader

class DocumentParser:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extracts raw text string from a local PDF document file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target PDF file not found at: {file_path}")
            
        text = ""
        reader = PdfReader(file_path)
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                
        return text.strip()

    @staticmethod
    def parse_sections(raw_text: str) -> dict:
        """
        Heuristically segments text into structural components: 
        Abstract, Methodology, and Budget markers.
        """
        sections = {
            "abstract": "",
            "methodology": "",
            "budget_details": "",
            "full_text": raw_text
        }
        
        lower_text = raw_text.lower()
        
  
        abstract_idx = lower_text.find("abstract")
        methodology_idx = lower_text.find("methodology")
        budget_idx = lower_text.find("budget")
        
        
        if abstract_idx != -1:
            end_idx = methodology_idx if methodology_idx != -1 else (budget_idx if budget_idx != -1 else len(raw_text))
            sections["abstract"] = raw_text[abstract_idx:end_idx].strip()
            
        if methodology_idx != -1:
            end_idx = budget_idx if budget_idx != -1 else len(raw_text)
            sections["methodology"] = raw_text[methodology_idx:end_idx].strip()
            
        if budget_idx != -1:
            sections["budget_details"] = raw_text[budget_idx:].strip()
            
        if not sections["abstract"] and not sections["methodology"]:
            sections["abstract"] = raw_text[:1000] # First 1000 chars as placeholder overview
            
        return sections