import re
import json
from typing import Tuple, Optional
from core.runner import create_llm
from langchain_openai import ChatOpenAI
from core.config import Settings
from langchain_core.messages import HumanMessage, SystemMessage

class Scorer:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        # Use a DIFFERENT model for judging than the one in .env to prevent self-grading bias
        # Typical Groq model for judging: llama-3.3-70b-versatile
        self.judge_model = "llama-3.3-70b-versatile"
        self.judge_llm = ChatOpenAI(
            model=self.judge_model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            temperature=0, # Deterministic judging
            max_tokens=400,
        )

    def extract_numeric_answer(self, prediction: str) -> str:
        """Uses LLM to extract JUST the numeric value from a long-form response."""
        prompt = (
            f"Extract only the specific numeric value (including currency symbols or units like 'billion') "
            f"that represents the final answer to the user query from this memo:\n\n"
            f"{prediction}\n\n"
            f"If there are multiple values, find the one that most closely matches the main topic of the query. "
            f"Return ONLY the value, no text."
        )
        try:
            res = self.judge_llm.invoke([
                SystemMessage(content="You are a data extraction assistant. Return ONLY the number."),
                HumanMessage(content=prompt)
            ])
            return res.content.strip()
        except:
            return prediction

    def normalize_numeric(self, text: str) -> Optional[float]:
        """Convert strings like '$94.9B', '1,200,000', '10.5 million' to a float."""
        if not text:
            return None
        
        # Remove currency symbols and commas
        text = text.replace("$", "").replace(",", "").strip()
        
        # Handle units
        multipliers = {
            "b": 1e9, "billion": 1e9,
            "m": 1e6, "million": 1e6,
            "k": 1e3, "thousand": 1e3
        }
        
        # Extract number and unit
        match = re.search(r"([-+]?\d*\.?\d+)\s*([a-zA-Z]*)", text, re.IGNORECASE)
        if not match:
            return None
            
        val = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit in multipliers:
            val *= multipliers[unit]
            
        return val

    def score_numeric(self, prediction: str, expected: str) -> float:
        """Returns 1.0 if match, 0.0 otherwise. Uses 1% tolerance."""
        # Extract short answer first if it's a long memo
        if len(prediction) > 50:
            prediction = self.extract_numeric_answer(prediction)
            
        pred_val = self.normalize_numeric(prediction)
        exp_val = self.normalize_numeric(expected)
        
        if pred_val is None or exp_val is None:
            return 0.0
            
        if exp_val == 0:
            return 1.0 if pred_val == 0 else 0.0
            
        error = abs(pred_val - exp_val) / abs(exp_val)
        return 1.0 if error < 0.01 else 0.0

    def score_qualitative(self, question: str, prediction: str, expected: str) -> Tuple[float, str]:
        """Uses LLM to judge if the prediction matches the expected fact."""
        prompt = f"""
        You are a strict financial evaluation judge. 
        Your task is to determine if the 'Agent Response' correctly contains the 'Expected Fact' for the given 'Question'.
        
        Question: {question}
        Expected Fact: {expected}
        Agent Response: {prediction}
        
        Rules:
        1. If the Agent Response contains the core expected fact and is factually consistent, score 1.0.
        2. If the Agent Response is partially correct or missing key details, score 0.5.
        3. If the Agent Response is wrong, contradicts the fact, or says 'I don't know', score 0.0.
        
        Return ONLY a JSON object with 'score' (float) and 'reason' (short string).
        """
        
        try:
            res = self.judge_llm.invoke([
                SystemMessage(content="You are a binary evaluation judge. Return JSON."),
                HumanMessage(content=prompt)
            ])
            # Handle potential markdown formatting in LLM response
            content = res.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return float(data.get("score", 0.0)), data.get("reason", "No reason provided.")
        except Exception as e:
            return 0.0, f"Scoring error: {str(e)}"

if __name__ == "__main__":
    # Quick test
    s = Scorer()
    print(f"Numeric Match ($94.9B vs 94900000000): {s.score_numeric('$94.9B', '94900000000')}")
    print(f"Qualitative Match: {s.score_qualitative('Risk of competition?', 'High competitive pressure from cloud rivals', 'Intense competition in cloud space')}")
