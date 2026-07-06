import re

def extract_budget_from_text(proposal_text: str):
    """
    Extracts budget numbers written in ₹, INR, or Rs format from proposal text,
    handling commas, decimals, and words like 'lakh'.
    """
    budget_values = []

    # This pattern catches: ₹500000, INR 5,00,000, Rs. 3.5 lakh, Rs 5 Lakhs
    pattern = r"(?:₹|INR|Rs\.?)\s?([\d,]+\.?\d*)\s?(lakh)?s?"

    # Find all matches (case-insensitive to catch 'lakh' or 'Lakh')
    matches = re.findall(pattern, proposal_text, re.IGNORECASE)

    for number_part, lakh_part in matches:
        #Clean the numeric string by removing commas
        clean_num = number_part.replace(",", "")
        
        try:
            val = float(clean_num)
            
            # 2. If the word 'lakh' was found, multiply by 100,000
            if lakh_part:
                val *= 100000
                
            budget_values.append(val)
        except ValueError:
            continue # Skip if something went wrong converting to float

    if not budget_values:
        return None

    return max(budget_values)