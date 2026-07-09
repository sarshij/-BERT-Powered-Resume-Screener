
# ENETR API KEY ON LINE 29

import os
import re
from abc import ABC, abstractmethod
from app.logger import setup_logger

logger = setup_logger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def evaluate_plausibility(self, resume_text: str, job_description: str) -> float:
        """
        Evaluate if a resume is likely AI-generated or factually inconsistent.
        Returns a float between 0.0 (Authentic/Plausible) and 1.0 (Highly Suspicious/AI).
        """
        pass

    @abstractmethod
    def verify_prediction(self, resume_text: str, job_description: str, local_classification: str) -> dict | None:
        """
        Double check the local model's prediction.
        Returns a dict: {"consensus": "Agree" | "Disagree", "reasoning": "..."} or None if failed.
        """
        pass

class GroqProvider(LLMProvider):
    def __init__(self):
        try:
            from groq import Groq
            api_key = os.environ.get("GROQ_API_KEY", "ENTER YOUR API KEY")
            self.client = Groq(api_key=api_key)
            self.model = "llama-3.3-70b-versatile"
            self.available = api_key != "dummy_key_for_tests"
        except ImportError:
            self.client = None
            self.available = False
            logger.warning("groq library not installed")

    def _sanitize(self, text: str) -> str:
        # Prevent basic prompt injection (e.g. system instruction overrides)
        text = re.sub(r'(?i)(ignore previous instructions|system prompt|you are a)', '', text)
        return text[:2000] # Limit length

    def evaluate_plausibility(self, resume_text: str, job_description: str) -> float | None:
        if not self.available or not self.client:
            return None

        prompt = f"""
        Analyze the following resume excerpt against the job description.
        Evaluate it strictly for AI-generation "tells" (extreme buzzword stuffing, unnatural perfection, lack of concrete metric variance) 
        and logical inconsistencies (e.g. leading architecture at a junior age).
        
        Resume: {self._sanitize(resume_text)}
        
        Job: {self._sanitize(job_description)}
        
        Return ONLY a single float between 0.0 and 1.0 representing the probability that this resume is AI-generated and factually exaggerated.
        (0.0 = very human and plausible, 1.0 = highly likely AI hallucination). Do not return any other text.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise AI text detector. Output only a float between 0.0 and 1.0."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            # Extract float
            match = re.search(r'0\.\d+|1\.0|0', result)
            if match:
                return float(match.group(0))
            return None
        except Exception as e:
            logger.error(f"⚠️ \033[1;33mGroq API evaluation failed: {e}\033[0m")
            return None

    def verify_prediction(self, resume_text: str, job_description: str, local_classification: str) -> dict | None:
        if not self.available or not self.client:
            return None

        prompt = f"""
        Our local AI model classified this resume as '{local_classification}'. Act as an expert HR auditor. Do you agree?
        
        Resume: {self._sanitize(resume_text)}
        Job Description: {self._sanitize(job_description)}
        
        Respond ONLY with a valid JSON object. Give reasoning in around 30-50 sweet words explaining exactly WHY it is suspicious or authentic:
        {{"consensus": "Agree", "reasoning": "<30-50 word explanation>"}}
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content.strip()
            import json
            data = json.loads(result)
            if "consensus" in data and "reasoning" in data:
                return data
            return None
        except Exception as e:
            logger.error(f"⚠️ \033[1;33mGroq API verification failed: {e}\033[0m")
            return None

class NvidiaProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
            api_key = os.environ.get("NVIDIA_API_KEY", "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")
            self.client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            self.model = "nvidia/nemotron-3-ultra-550b-a55b"
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            logger.warning("openai library not installed")

    def _sanitize(self, text: str) -> str:
        text = re.sub(r'(?i)(ignore previous instructions|system prompt|you are a)', '', text)
        return text[:2000]

    def evaluate_plausibility(self, resume_text: str, job_description: str) -> float | None:
        if not self.available or not self.client:
            return None

        prompt = f"""
        Analyze the following resume excerpt against the job description.
        Evaluate it strictly for AI-generation "tells" (extreme buzzword stuffing, unnatural perfection, lack of concrete metric variance) 
        and logical inconsistencies (e.g. leading architecture at a junior age).
        
        Resume: {self._sanitize(resume_text)}
        
        Job: {self._sanitize(job_description)}
        
        Return ONLY a single float between 0.0 and 1.0 representing the probability that this resume is AI-generated and factually exaggerated.
        (0.0 = very human and plausible, 1.0 = highly likely AI hallucination). Do not return any other text.
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": "You are a precise AI text detector. Output only a float between 0.0 and 1.0. " + prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip()
            match = re.search(r'0\.\d+|1\.0|0', result)
            if match:
                return float(match.group(0))
            return None
        except Exception as e:
            logger.error(f"⚠️ \033[1;33mNvidia API evaluation failed: {e}\033[0m")
            return None

    def verify_prediction(self, resume_text: str, job_description: str, local_classification: str) -> dict | None:
        if not self.available or not self.client:
            return None

        prompt = f"""
        Our local AI model classified this resume as '{local_classification}'. Act as an expert HR auditor. Do you agree?
        
        Resume: {self._sanitize(resume_text)}
        Job Description: {self._sanitize(job_description)}
        
        Respond ONLY with a valid JSON object. Give reasoning in around 30-50 sweet words explaining exactly WHY it is suspicious or authentic:
        {{"consensus": "Agree", "reasoning": "<30-50 word explanation>"}}
        """
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content.strip()
            import json
            data = json.loads(result)
            # Ensure keys exist
            if "consensus" in data and "reasoning" in data:
                return data
            return None
        except Exception as e:
            logger.error(f"⚠️ \033[1;33mNvidia API verification failed: {e}\033[0m")
            return None

class FallbackProvider(LLMProvider):
    def __init__(self):
        # Try Groq first, then Nvidia
        self.providers = [GroqProvider(), NvidiaProvider()]

    def evaluate_plausibility(self, resume_text: str, job_description: str) -> float:
        for i, provider in enumerate(self.providers):
            if provider.available:
                if i == 0:
                    logger.info(f"🤖 \033[1;36m[LLM Check: Triggering {provider.__class__.__name__}]\033[0m")
                else:
                    logger.info(f"🔄 \033[1;35m[Multi-LLM Fallback: Triggering {provider.__class__.__name__}]\033[0m")
                score = provider.evaluate_plausibility(resume_text, job_description)
                if score is not None:
                    return score
        return 0.5 # Ultimate fallback if all APIs fail

    def verify_prediction(self, resume_text: str, job_description: str, local_classification: str) -> dict | None:
        for i, provider in enumerate(self.providers):
            if provider.available:
                if i == 0:
                    logger.info(f"🤖 \033[1;36m[LLM Check: Triggering {provider.__class__.__name__}]\033[0m")
                else:
                    logger.info(f"🔄 \033[1;35m[Multi-LLM Fallback: Triggering {provider.__class__.__name__}]\033[0m")
                result = provider.verify_prediction(resume_text, job_description, local_classification)
                if result is not None:
                    return result
        return None

# Factory
_fallback_provider_instance = None

def get_llm_detector() -> LLMProvider:
    global _fallback_provider_instance
    if _fallback_provider_instance is None:
        _fallback_provider_instance = FallbackProvider()
    return _fallback_provider_instance
