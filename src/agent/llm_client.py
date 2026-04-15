# src/agent/llm_client.py
"""
FIXED: Enhanced JSON extraction that handles markdown code fences and malformed responses.
"""

import json
import re
import time
from typing import Optional, Dict, Any

OLLAMA_API_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"


def _extract_first_json_segment(text: str) -> Optional[str]:
    """
    Enhanced extraction that handles markdown code fences.
    """
    if not text:
        return None
    
    # Step 1: Remove markdown code fences
    s = text.strip()
    
    # Remove ```json ... ``` or ``` ... ```
    s = re.sub(r'```json\s*', '', s)
    s = re.sub(r'```\s*', '', s)
    s = s.strip()
    
    # Step 2: Try to find complete JSON via regex
    matches = re.findall(r'(\{[\s\S]*?\}(?=\s*(?:\{|$))|\[[\s\S]*?\](?=\s*(?:\[|$)))', s)
    for seg in matches:
        seg = seg.strip()
        try:
            json.loads(seg)
            return seg
        except Exception:
            pass

    # Step 3: Manual bracket matching for first balanced object/array
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = None
        for i, ch in enumerate(s):
            if ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    cand = s[start:i + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        # continue searching
                        pass

    # Step 4: Try first { ... last } or [ ... last ]
    i = s.find("{")
    j = s.rfind("}")
    if 0 <= i < j:
        cand = s[i:j + 1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass
    
    i = s.find("[")
    j = s.rfind("]")
    if 0 <= i < j:
        cand = s[i:j + 1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            pass

    return None


def _salvage_json_from_text(text: str) -> Optional[str]:
    """
    Conservative salvage: attempt to trim trailing prose until braces balance.
    """
    if not text:
        return None
    
    # Remove markdown first
    s = re.sub(r'```json\s*', '', text)
    s = re.sub(r'```\s*', '', s)
    s = s.strip()
    
    first_obj = min([pos for pos in [s.find("{"), s.find("[")] if pos != -1], default=-1)
    if first_obj == -1:
        return None
    
    # Find last potential close
    last_close = max(s.rfind("}"), s.rfind("]"))
    if last_close <= first_obj:
        return None
    
    # Try decreasing end until balanced braces
    for end in range(last_close, first_obj, -1):
        cand = s[first_obj:end + 1]
        # Quick balance heuristic
        if cand.count("{") != cand.count("}"):
            continue
        if cand.count("[") != cand.count("]"):
            continue
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    
    return None


class HuggingFaceLLMClient:
    """
    Direct HuggingFace LLM client - loads model once and reuses it.
    """
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or DEFAULT_MODEL
        self.tokenizer = None
        self.hf_model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the HuggingFace model once."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[HF-LLM] Loading {self.model} to {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation="eager"  # Avoid flash attention warnings
            )
            print(f"[HF-LLM] Model loaded successfully")
        except Exception as e:
            print(f"[HF-LLM] Failed to load model: {e}")
            raise

    def _http_post(self, payload: Dict[str, Any], timeout: int = 60) -> str:
        """Compatibility method - now uses HuggingFace directly."""
        prompt = payload.get("prompt", "")
        max_tokens = payload.get("max_tokens", 512)
        temperature = payload.get("temperature", 0.0)
        
        return self._hf_generate(prompt, max_tokens, temperature)
    
    def _hf_generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate text using HuggingFace model."""
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.01,
                    do_sample=temperature > 0,
                    use_cache=True,  # Enable KV caching
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return generated_text
            
        except Exception as e:
            print(f"[HF-LLM] Generation failed: {e}")
            return ""

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Optional[str]:
        """Return raw text generation."""
        return self._hf_generate(prompt, max_tokens, temperature)

    # Keep the existing generate_json method unchanged - it calls _http_post which now uses HF


    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Optional[str]:
        """
        Return raw text generation.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        }
        raw = self._http_post(payload)
        return raw

    def generate_json(self, system_hint: Optional[str], user_prompt: Optional[str], schema_text: Optional[str],
                      max_tokens: int = 512, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        FIXED: Enhanced JSON generation with markdown fence removal.
        Returns parsed JSON object or diagnostic dict on parse failure.
        """
        # Build a compact instruction requesting only JSON
        sys_hint = (system_hint or "Output ONLY valid JSON that strictly matches the requested schema. No preamble, no markdown, no code fences.") + "\n"
        composed = f"{sys_hint}\nUser prompt:\n{user_prompt or ''}\n\nSchema:\n{schema_text or ''}\nRespond with a single JSON object only. No markdown code fences."
        
        payload = {
            "model": self.model,
            "prompt": composed,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        }
        raw = self._http_post(payload)
        
        if raw is None:
            raw = ""
        
        # Debug: Print first 200 chars of raw response
        preview = (raw or "")[:200].replace("\n", " ")
        if "```" in raw[:50]:
            print(f"[LLM][JSON] Warning: Response contains markdown fences: {preview}")
        
        # Try to extract JSON substring (with markdown fence removal)
        seg = _extract_first_json_segment(raw)
        if seg:
            try:
                obj = json.loads(seg)
                # Success - return parsed object
                return obj
            except Exception as e:
                print(f"[LLM][JSON] Parse error after extraction: {e}")
                # Attempt salvage
                salv = _salvage_json_from_text(raw)
                if salv:
                    try:
                        obj = json.loads(salv)
                        print("[LLM][JSON] Salvage successful")
                        return obj
                    except Exception:
                        return {"__status": "PARSE_ERROR", "raw": raw[:4096], "salvaged": None}
                return {"__status": "PARSE_ERROR", "raw": raw[:4096], "salvaged": None}
        
        # No JSON seg found; attempt salvage
        salv = _salvage_json_from_text(raw)
        if salv:
            try:
                obj = json.loads(salv)
                print("[LLM][JSON] Salvage successful after no segment found")
                return obj
            except Exception:
                return {"__status": "PARSE_ERROR", "raw": raw[:4096], "salvaged": None}
        
        # Nothing salvageable
        print(f"[LLM][JSON] Complete parse failure. Raw preview: {preview}")
        return {"__status": "PARSE_ERROR", "raw": raw[:4096], "salvaged": None}
    
    # Compatibility alias so existing code doesn't break
OllamaClient = HuggingFaceLLMClient
