# src/agent/executor.py
"""
Executor with Self-Refinement Loop + Chain-of-Thought with Verification.
Maintains all existing skill cache and exemplar functionality.
"""
import json
import re
import time
import hashlib
from typing import Any, Dict, List, Optional
from .llm_client import OllamaClient

# Lazy loaded components
_INDEXSTORE = None
_VERIFIER = None
_CHOICE = None

def _lazy_indexstore(cache_dir: Optional[str]):
    global _INDEXSTORE
    if _INDEXSTORE is None:
        try:
            from src.skill_cache.index_store import IndexStore
            _INDEXSTORE = IndexStore(cache_dir or "./cache")
        except Exception:
            _INDEXSTORE = None
    return _INDEXSTORE

def _lazy_verifier():
    global _VERIFIER
    if _VERIFIER is None:
        try:
            from src.verifier.verifier import Verifier
            _VERIFIER = Verifier()
        except Exception:
            _VERIFIER = None
    return _VERIFIER

def _lazy_choice():
    global _CHOICE
    if _CHOICE is None:
        try:
            from src.tools.choice_selector import ChoiceSelector
            _CHOICE = ChoiceSelector()
        except Exception:
            _CHOICE = None
    return _CHOICE

def _normalize_text(s: str) -> str:
    if not s:
        return ""
    t = " ".join(s.strip().split())
    return t

def _problem_sha(text: str) -> str:
    n = _normalize_text(text)
    return hashlib.sha256(n.encode("utf-8")).hexdigest()

class ToolDispatcher:
    def __init__(self, python_tool=None, retriever=None, sympy_tool=None, z3_tool=None):
        self.python_tool = python_tool
        self.retriever = retriever
        self.sympy_tool = sympy_tool
        self.z3_tool = z3_tool

    def run(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "python":
                if self.python_tool:
                    out = self.python_tool.execute(**(args or {}))
                    return {"tool": "python", "ok": True, "data": out, "error": ""}
                code = args.get("code", "")
                if re.match(r'^[\d\.\+\-\*\/\(\)\s,]+$', code.strip()):
                    try:
                        val = eval(code, {"__builtins__": {}}, {})
                        return {"tool": "python", "ok": True, "data": {"result": val}, "error": ""}
                    except Exception as e:
                        return {"tool": "python", "ok": False, "data": {}, "error": str(e)}
                return {"tool": "python", "ok": False, "data": {}, "error": "unsafe-code"}
            elif tool_name == "retriever":
                if self.retriever:
                    out = self.retriever.execute(problem=args.get("problem",""), topic=args.get("topic",""), k=args.get("k",3))
                    return {"tool": "retriever", "ok": True, "data": out, "error": ""}
                return {"tool": "retriever", "ok": False, "data": {}, "error": "no-retriever"}
            elif tool_name == "sympy":
                if self.sympy_tool:
                    out = self.sympy_tool.execute(**(args or {}))
                    return {"tool": "sympy", "ok": True, "data": out, "error": ""}
                return {"tool": "sympy", "ok": False, "data": {}, "error": "no-sympy"}
            elif tool_name == "z3":
                if self.z3_tool:
                    out = self.z3_tool.execute(**(args or {}))
                    return {"tool": "z3", "ok": True, "data": out, "error": ""}
                return {"tool": "z3", "ok": False, "data": {}, "error": "no-z3"}
            elif tool_name == "choice_selector":
                chooser = _lazy_choice()
                if chooser:
                    out = chooser.execute(args.get("computed_result"), args.get("options", []))
                    return {"tool": "choice_selector", "ok": True, "data": out, "error": ""}
                return {"tool": "choice_selector", "ok": False, "data": {}, "error": "no-choice-selector"}
            else:
                return {"tool": tool_name, "ok": False, "data": {}, "error": "unknown-tool"}
        except Exception as e:
            return {"tool": tool_name, "ok": False, "data": {}, "error": str(e)}

class Executor:
    def __init__(self, llm: Optional[OllamaClient] = None, tools: Optional[ToolDispatcher] = None, cache_dir: Optional[str] = None):
        self.llm = llm or OllamaClient()
        self.tools = tools or ToolDispatcher()
        self.max_steps = 6
        self.cache_dir = cache_dir or "./cache"
        self.last_exemplars = None
        # NEW: Refinement settings
        self.enable_refinement = True
        self.max_refinement_iterations = 2
        self.enable_cot_verification = True

    def _goals_schema(self) -> str:
        return json.dumps({
            "type": "object",
            "properties": {"goals": {"type": "array", "items": {"type": "string"}, "maxItems": 5}},
            "required": ["goals"]
        })

    def _action_schema(self) -> str:
        return json.dumps({
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["TOOL", "FINAL"]},
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "final": {"type": "object", "properties": {
                    "answer": {"type": "string"}, 
                    "reasoning_summary": {"type": "string"}
                }}
            },
            "required": ["action_type"]
        })

    # NEW: Chain-of-Thought schema
    def _cot_schema(self) -> str:
        return json.dumps({
            "type": "object",
            "properties": {
                "reasoning_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step-by-step reasoning"
                },
                "final_answer": {"type": "string"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["reasoning_steps", "final_answer"]
        })

    # NEW: Critique schema
    def _critique_schema(self) -> str:
        return json.dumps({
            "type": "object",
            "properties": {
                "flaws": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Identified weaknesses or errors"
                },
                "improvements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested improvements"
                },
                "should_revise": {"type": "boolean"}
            },
            "required": ["flaws", "should_revise"]
        })

    def _clean_json_response(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        brace_pos = text.find('{')
        bracket_pos = text.find('[')
        if brace_pos == -1 and bracket_pos == -1:
            return text
        if brace_pos == -1:
            json_start = bracket_pos
        elif bracket_pos == -1:
            json_start = brace_pos
        else:
            json_start = min(brace_pos, bracket_pos)
        if text[json_start] == '{':
            depth = 0
            for i in range(json_start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return text[json_start:i+1]
        elif text[json_start] == '[':
            depth = 0
            for i in range(json_start, len(text)):
                if text[i] == '[':
                    depth += 1
                elif text[i] == ']':
                    depth -= 1
                    if depth == 0:
                        return text[json_start:i+1]
        return text

    def _llm_generate_json(self, system_hint: str, user_prompt: str, schema_text: str):
        try:
            if hasattr(self.llm, "generate_json"):
                result = self.llm.generate_json(
                    system_hint=system_hint,
                    user_prompt=user_prompt,
                    schema_text=schema_text,
                    max_tokens=512,
                    temperature=0.0
                )
                return result
            else:
                raw = self.llm.generate(
                    f"{system_hint}\n\n{user_prompt}",
                    max_tokens=512,
                    temperature=0.0
                )
                cleaned = self._clean_json_response(raw)
                try:
                    return json.loads(cleaned)
                except Exception:
                    match = re.search(r'\{[\s\S]*\}', cleaned)
                    if match:
                        try:
                            return json.loads(match.group(0))
                        except Exception:
                            pass
                try:
                    m = re.search(r'(\{[\s\S]*\})', raw)
                    if m:
                        try:
                            return json.loads(m.group(1))
                        except Exception:
                            return None
                except Exception:
                    pass
                return None
        except Exception:
            return None

    def _format_exemplar_qa(self, record: dict) -> dict:
        """Format a database record as Q&A exemplar."""
        solution = record.get("solution") or {}
        problem_text = record.get("problem_text", "")
        topic = record.get("topic", "")
        similarity = record.get("similarity", 0.0)

        final_answer = ""
        if "final_answer" in solution and solution["final_answer"] is not None:
            final_answer = str(solution["final_answer"])
        elif "final_answer_text" in solution and solution["final_answer_text"]:
            final_answer = str(solution["final_answer_text"])
        elif "reasoning_summary" in solution and solution["reasoning_summary"]:
            final_answer = str(solution["reasoning_summary"])

        print(f"[Executor._format_exemplar_qa] Extracted answer: '{final_answer[:50]}' from solution keys: {list(solution.keys())}")

        return {
            "topic": topic,
            "problem": problem_text,
            "answer": final_answer,
            "similarity": similarity
        }

    def _match_solution_to_options(self, record: dict, test_options: List[str]) -> Optional[int]:
        """Match using correct_option_number from train data."""
        solution_dict = record.get("solution") or {}
        if not record.get("correct_option_number") and "correct_option_number" in solution_dict:
            record["correct_option_number"] = solution_dict.get("correct_option_number")

        if not any(k.startswith("answer_option_") for k in record.keys()):
            for i, opt in enumerate(solution_dict.get("options", []) or [], 1):
                record[f"answer_option_{i}"] = opt

        correct_option_num = record.get("correct_option_number")
        if correct_option_num is not None:
            try:
                train_option_idx = int(correct_option_num)
                train_option_text = None
                for j in range(1, 20):
                    option_key = f"answer_option_{j}"
                    if option_key in record:
                        if j == train_option_idx:
                            train_option_text = record.get(option_key)
                            break

                if train_option_text:
                    print(f"[Executor._match] Train correct_option_number={train_option_idx}, text='{train_option_text[:60]}'")
                    for i, test_opt in enumerate(test_options, 1):
                        if test_opt.strip().lower() == train_option_text.strip().lower():
                            print(f"[Executor._match] ✅ Exact text match with test option {i}")
                            return i

                    best_match = None
                    best_score = 0.0
                    train_words = set(train_option_text.lower().split())
                    for i, test_opt in enumerate(test_options, 1):
                        test_words = set(test_opt.lower().split())
                        common = train_words & test_words
                        score = len(common) / max(len(train_words), len(test_words), 1)
                        if score > best_score:
                            best_score = score
                            best_match = i

                    if best_score >= 0.5:
                        print(f"[Executor._match] ✅ Semantic match with test option {best_match}, score={best_score:.2f}")
                        return best_match
            except Exception as e:
                print(f"[Executor._match] Error matching correct_option_number: {e}")

        final_answer = solution_dict.get("final_answer", "")
        if not final_answer:
            final_answer = solution_dict.get("final_answer_text", "")

        reasoning = solution_dict.get("reasoning_summary", "") or solution_dict.get("final_answer_text", "")
        solution_text = f"{final_answer} {reasoning}".lower()

        try:
            idx = int(final_answer)
            if 1 <= idx <= len(test_options):
                print(f"[Executor._match] Exact int match from final_answer: {idx}")
                return idx
        except Exception:
            pass

        best_match = None
        best_score = 0.0
        for i, option in enumerate(test_options, 1):
            option_lower = option.lower()
            if option_lower in solution_text or solution_text in option_lower:
                common_words = set(option_lower.split()) & set(solution_text.split())
                score = len(common_words) / max(len(option_lower.split()), 1)
                if score > best_score:
                    best_score = score
                    best_match = i

        if best_score > 0.5:
            print(f"[Executor._match] Solution text semantic match: option={best_match}, score={best_score:.2f}")
            return best_match

        chooser = self.tools.run("choice_selector", {
            "computed_result": solution_text,
            "options": test_options
        })
        if chooser.get("ok") and chooser.get("data", {}).get("success"):
            sel = int(chooser["data"]["data"]["selected_index"])
            print(f"[Executor._match] Choice selector match: {sel}")
            return sel

        return None

    def _ask_goals(self, problem: str, topic: str, options: List[str], exemplars: List[dict] = None) -> List[str]:
        system_hint = f"You are an expert problem solver in {topic}. Produce only a JSON object matching the schema. Keep goals short."
        exemplar_text = ""
        if exemplars:
            ex_lines = []
            for i, ex in enumerate(exemplars, 1):
                ex_lines.append(f"Q: Topic: {ex.get('topic', '')}")
                ex_lines.append(f"Problem: {ex.get('problem', '')}")
                ex_lines.append(f"A: {ex.get('answer', '')}")
                ex_lines.append("")
            exemplar_text = "\n".join(ex_lines) + "\n"
            print(f"[Executor._ask_goals] Injected {len(exemplars)} exemplar(s) into planner prompt")
        user_prompt = f"{exemplar_text}Q: Topic: {topic}\nProblem: {problem}\nChoices: {options}\nA:\n\nProduce up to 5 short goals (each <= 10 words) as JSON {{'goals': [...] }}."
        schema = self._goals_schema()
        try:
            obj = self._llm_generate_json(system_hint, user_prompt, schema)
            if isinstance(obj, dict) and "goals" in obj and isinstance(obj["goals"], list):
                goals = [(" ".join(g.strip().split()))[:120] for g in obj["goals"]][:5]
                return goals
        except Exception:
            pass
        return []

    # NEW: Chain-of-Thought reasoning
    def _cot_reasoning(self, problem: str, topic: str, options: List[str], exemplars: List[dict] = None) -> Dict[str, Any]:
        """Generate step-by-step reasoning with explicit Chain-of-Thought."""
        system_hint = "You are a systematic problem solver. Show your step-by-step reasoning explicitly."
        
        exemplar_text = ""
        if exemplars:
            ex_lines = []
            for ex in exemplars[:2]:
                ex_lines.append(f"Example: {ex.get('problem', '')[:200]}")
                ex_lines.append(f"Answer: {ex.get('answer', '')}")
            exemplar_text = "\n".join(ex_lines) + "\n\n"
        
        user_prompt = f"""{exemplar_text}Problem: {problem}
Topic: {topic}
Options: {', '.join([f"({i+1}) {opt}" for i, opt in enumerate(options)])}

Provide your reasoning in clear steps, then state your final answer.
Response format (JSON):
{{
  "reasoning_steps": ["Step 1: ...", "Step 2: ...", ...],
  "final_answer": "your answer",
  "confidence": "high/medium/low"
}}"""
        
        schema = self._cot_schema()
        obj = self._llm_generate_json(system_hint, user_prompt, schema)
        
        if obj and isinstance(obj, dict):
            print(f"[CoT] Generated {len(obj.get('reasoning_steps', []))} reasoning steps")
            return obj
        return {"reasoning_steps": [], "final_answer": "", "confidence": "low"}

    # NEW: Self-critique
    def _critique_solution(self, problem: str, topic: str, options: List[str], solution: Dict[str, Any]) -> Dict[str, Any]:
        """Critique a proposed solution to identify flaws."""
        system_hint = "You are a critical reviewer. Identify flaws, errors, and areas for improvement."
        
        steps = solution.get("reasoning_steps", [])
        answer = solution.get("final_answer", "")
        
        user_prompt = f"""Problem: {problem}
Topic: {topic}
Options: {', '.join([f"({i+1}) {opt}" for i, opt in enumerate(options)])}

Proposed Solution:
Reasoning: {' -> '.join(steps)}
Answer: {answer}

Critically analyze this solution. Identify any flaws, logical errors, or missed considerations.
Should this solution be revised?

Response format (JSON):
{{
  "flaws": ["flaw 1", "flaw 2", ...],
  "improvements": ["improvement 1", ...],
  "should_revise": true/false
}}"""
        
        schema = self._critique_schema()
        obj = self._llm_generate_json(system_hint, user_prompt, schema)
        
        if obj and isinstance(obj, dict):
            print(f"[Critique] Found {len(obj.get('flaws', []))} flaws, should_revise={obj.get('should_revise', False)}")
            return obj
        return {"flaws": [], "improvements": [], "should_revise": False}

    # NEW: Refined reasoning
    def _refined_reasoning(self, problem: str, topic: str, options: List[str], 
                          original_solution: Dict[str, Any], critique: Dict[str, Any],
                          exemplars: List[dict] = None) -> Dict[str, Any]:
        """Generate improved solution based on critique."""
        system_hint = "You are a problem solver improving a previous solution based on feedback."
        
        exemplar_text = ""
        if exemplars:
            ex_lines = []
            for ex in exemplars[:2]:
                ex_lines.append(f"Example: {ex.get('problem', '')[:200]}")
                ex_lines.append(f"Answer: {ex.get('answer', '')}")
            exemplar_text = "\n".join(ex_lines) + "\n\n"
        
        user_prompt = f"""{exemplar_text}Problem: {problem}
Topic: {topic}
Options: {', '.join([f"({i+1}) {opt}" for i, opt in enumerate(options)])}

Previous Solution:
{' -> '.join(original_solution.get('reasoning_steps', []))}
Answer: {original_solution.get('final_answer', '')}

Identified Issues:
{chr(10).join(['- ' + f for f in critique.get('flaws', [])])}

Improvements Needed:
{chr(10).join(['- ' + i for i in critique.get('improvements', [])])}

Provide an IMPROVED solution addressing these issues.

Response format (JSON):
{{
  "reasoning_steps": ["Step 1: ...", "Step 2: ...", ...],
  "final_answer": "your improved answer",
  "confidence": "high/medium/low"
}}"""
        
        schema = self._cot_schema()
        obj = self._llm_generate_json(system_hint, user_prompt, schema)
        
        if obj and isinstance(obj, dict):
            print(f"[Refined] Generated improved solution with {len(obj.get('reasoning_steps', []))} steps")
            return obj
        return original_solution

    # NEW: Self-refinement loop
    def _solve_with_refinement(self, problem: str, topic: str, options: List[str], exemplars: List[dict] = None) -> Dict[str, Any]:
        """Solve with iterative self-refinement."""
        audit = []
        
        # Initial CoT reasoning
        print("\n[Refinement] === Initial CoT Reasoning ===")
        solution = self._cot_reasoning(problem, topic, options, exemplars)
        audit.append("cot_initial")
        
        if not self.enable_refinement:
            audit.append("refinement_disabled")
            return {"solution": solution, "audit": audit}
        
        # Refinement loop
        for iteration in range(self.max_refinement_iterations):
            print(f"\n[Refinement] === Iteration {iteration + 1}/{self.max_refinement_iterations} ===")
            
            # Critique
            critique = self._critique_solution(problem, topic, options, solution)
            audit.append(f"critique_iter_{iteration+1}_flaws_{len(critique.get('flaws', []))}")
            
            # Check if revision needed
            if not critique.get("should_revise", False):
                print(f"[Refinement] No revision needed after iteration {iteration + 1}")
                audit.append(f"refinement_converged_iter_{iteration+1}")
                break
            
            if len(critique.get("flaws", [])) == 0:
                print(f"[Refinement] No flaws found, stopping refinement")
                audit.append(f"no_flaws_iter_{iteration+1}")
                break
            
            # Generate refined solution
            solution = self._refined_reasoning(problem, topic, options, solution, critique, exemplars)
            audit.append(f"refined_iter_{iteration+1}")
        else:
            audit.append("refinement_max_iterations")
        
        return {"solution": solution, "audit": audit}

    # NEW: Verify CoT steps
    def _verify_cot_steps(self, steps: List[str], problem: str, options: List[str]) -> Dict[str, Any]:
        """Verify each reasoning step for logical consistency."""
        if not self.enable_cot_verification:
            return {"verified": True, "invalid_steps": []}
        
        invalid_steps = []
        for i, step in enumerate(steps):
            # Simple heuristic checks
            if len(step.strip()) < 10:
                invalid_steps.append({"step_num": i+1, "reason": "too_short"})
            # Add more verification logic as needed
        
        print(f"[CoT Verify] Checked {len(steps)} steps, found {len(invalid_steps)} invalid")
        return {"verified": len(invalid_steps) == 0, "invalid_steps": invalid_steps}

    def _llm_action(self, problem, topic, options, history, exemplars, goals=None):
        system_hint = "Output ONLY valid JSON that matches the schema. Respond with a single JSON object."
        exemplar_text = ""
        if exemplars:
            ex_lines = []
            for i, ex in enumerate(exemplars, 1):
                ex_lines.append(f"Q: Topic: {ex.get('topic', '')}")
                ex_lines.append(f"Problem: {ex.get('problem', '')}")
                ex_lines.append(f"A: {ex.get('answer', '')}")
                ex_lines.append("")
            exemplar_text = "\n".join(ex_lines) + "\n"
            print(f"[Executor._llm_action] Injected {len(exemplars)} exemplar(s) into action prompt")
        
        # NEW: Include goals in prompt
        goals_text = ""
        if goals:
            goals_text = "Goals:\n" + "\n".join([f"- {g}" for g in goals]) + "\n\n"
        
        history_text = ""
        if history:
            history_text = "\n".join([f"Observation: {h}" for h in history]) + "\n"
        user_prompt = f"{exemplar_text}{goals_text}Q: Topic: {topic}\nProblem: {problem}\nChoices: {options}\n{history_text}A:\n\nDecide next action."
        schema = self._action_schema()
        obj = self._llm_generate_json(system_hint, user_prompt, schema)
        return obj

    def _extract_answer_from_text(self, text: str, options: List[str]) -> Optional[int]:
        if not text:
            return None
        nums = re.findall(r'\b([1-9]|10)\b', text)
        if nums:
            idx = int(nums[0])
            if 1 <= idx <= len(options):
                return idx
        ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
        text_lower = text.lower()
        for word, num in ordinals.items():
            if word in text_lower and num <= len(options):
                return num
        chooser = self.tools.run("choice_selector", {
            "computed_result": text,
            "options": options
        })
        if chooser.get("ok") and chooser.get("data", {}).get("success"):
            return int(chooser["data"]["data"]["selected_index"])
        return None

    def solve(self, problem: str, topic: str, options: List[str]) -> Dict[str, Any]:
        audit: List[str] = []
        idx_store = _lazy_indexstore(self.cache_dir)
        verifier = _lazy_verifier()

        injected_exemplars = getattr(self, 'last_exemplars', None)
        if injected_exemplars:
            print(f"[Executor.solve] Found {len(injected_exemplars)} exemplar(s) injected by run_batch")

        # Exact lookup (unchanged)
        sha = _problem_sha(problem)
        try:
            if idx_store:
                exact = idx_store.exact_lookup(sha)
                if exact:
                    sol = exact.get("solution") or exact.get("solution_json") or {}
                    if verifier and sol:
                        verify_input = {
                            "problem_text": problem,
                            "topic": topic,
                            "parsed_constraints": exact.get("constraints", {}),
                            "candidate": {"final_answer": sol.get("final_answer", ""), "audit_trail": ["skill_cache_exact"]}
                        }
                        vout = verifier.verify({**verify_input, "options": options})
                        audit.append(f"skill_cache_exact_hit")
                        if vout.get("status") == "PASS":
                            norm = vout.get("normalized_answer") or sol.get("final_answer")
                            try:
                                sel = int(norm)
                                if 1 <= sel <= len(options):
                                    return {"selected_option": sel, "solution_text": sol, "audit_trail": audit}
                            except Exception:
                                chooser = self.tools.run("choice_selector", {"computed_result": norm, "options": options})
                                if chooser.get("ok"):
                                    sel = chooser.get("data",{}).get("data",{}).get("selected_index",1)
                                    return {"selected_option": int(sel), "solution_text": sol, "audit_trail": audit}
        except Exception as e:
            audit.append(f"exact_lookup_err:{e}")

        # Use injected exemplars (unchanged)
        exemplars = []
        if injected_exemplars and len(injected_exemplars) > 0:
            top_rec = injected_exemplars[0]
            sim = top_rec.get("similarity", 0.0)
            audit.append(f"retrieved_similar_top_sim={sim:.4f}")
            print(f"[Executor.solve] Top exemplar similarity: {sim:.4f}")

            if sim >= 0.80:
                audit.append(f"high_confidence_sim={sim:.4f}_attempting_direct_match")
                print(f"[Executor.solve] High confidence (>= 0.80), attempting direct option match...")

                matched_option = self._match_solution_to_options(top_rec, options)
                if matched_option is not None:
                    audit.append(f"high_confidence_match_found_option={matched_option}_skipped_llm")
                    print(f"[Executor.solve] ✅ Direct match found! Option {matched_option}, skipping LLM")
                    return {
                        "selected_option": matched_option,
                        "solution_text": top_rec.get("solution", {}),
                        "audit_trail": audit
                    }
                else:
                    audit.append(f"high_confidence_no_option_match_injecting_as_fewshot")
                    print(f"[Executor.solve] No direct match, will inject as few-shot example")
                    exemplars = [self._format_exemplar_qa(top_rec)]
                    audit.append(f"few_shot_single_exemplar_sim={sim:.4f}")
            else:
                print(f"[Executor.solve] Similarity < 0.80, injecting {len(injected_exemplars)} exemplars as few-shot")
                for r in injected_exemplars:
                    exemplars.append(self._format_exemplar_qa(r))
                audit.append(f"few_shot_injected_count={len(exemplars)}")

        # Planner step
        goals = self._ask_goals(problem, topic, options, exemplars)
        if goals:
            audit.append(f"goals_emitted:{len(goals)}")
        else:
            audit.append("no_goals")

        # NEW: Use refinement-based solving if enabled
        if self.enable_refinement or self.enable_cot_verification:
            print("\n[Executor] Using refinement-based solving...")
            refinement_result = self._solve_with_refinement(problem, topic, options, exemplars)
            audit.extend(refinement_result.get("audit", []))
            
            solution = refinement_result.get("solution", {})
            final_answer = solution.get("final_answer", "")
            reasoning_steps = solution.get("reasoning_steps", [])
            
            # Verify CoT steps
            if self.enable_cot_verification and reasoning_steps:
                verification = self._verify_cot_steps(reasoning_steps, problem, options)
                if verification.get("verified"):
                    audit.append("cot_steps_verified")
                else:
                    audit.append(f"cot_verification_failed_invalid_steps_{len(verification.get('invalid_steps', []))}")
            
            # Extract answer from refined solution
            extracted = self._extract_answer_from_text(final_answer, options)
            if extracted:
                audit.append("refinement_answer_extracted")
                return {
                    "selected_option": extracted,
                    "solution_text": {
                        "answer": final_answer,
                        "reasoning_steps": reasoning_steps,
                        "confidence": solution.get("confidence", "medium")
                    },
                    "audit_trail": audit
                }
            
            # Try verifier on refined answer
            if verifier:
                try:
                    verify_input = {
                        "problem_text": problem,
                        "topic": topic,
                        "parsed_constraints": {},
                        "candidate": {"final_answer": final_answer, "audit_trail": reasoning_steps},
                        "options": options
                    }
                    vout = verifier.verify(verify_input)
                    audit.append(f"verifier_refinement:{vout.get('status')}")
                    if vout.get("status") == "PASS":
                        normalized = vout.get("normalized_answer") or final_answer
                        try:
                            selected_option = int(str(normalized).strip())
                            if 1 <= selected_option <= len(options):
                                return {
                                    "selected_option": selected_option,
                                    "solution_text": {
                                        "answer": final_answer,
                                        "reasoning_steps": reasoning_steps
                                    },
                                    "audit_trail": audit
                                }
                        except Exception:
                            pass
                        
                        # Choice selector on normalized
                        if options:
                            chooser = self.tools.run("choice_selector", {
                                "computed_result": str(normalized),
                                "options": options
                            })
                            if chooser.get("ok") and chooser.get("data", {}).get("success"):
                                sel = int(chooser["data"]["data"]["selected_index"])
                                return {
                                    "selected_option": sel,
                                    "solution_text": {
                                        "answer": final_answer,
                                        "reasoning_steps": reasoning_steps
                                    },
                                    "audit_trail": audit
                                }
                except Exception as e:
                    audit.append(f"verifier_refinement_exception:{e}")
            
            # Fallback: choice selector on refined answer
            if options:
                audit.append("refinement_fallback_choice_selector")
                reasoning_text = " ".join(reasoning_steps[:3])
                chooser = self.tools.run("choice_selector", {
                    "computed_result": f"{final_answer} {reasoning_text}",
                    "options": options
                })
                if chooser.get("ok") and chooser.get("data", {}).get("success"):
                    sel = int(chooser["data"]["data"]["selected_index"])
                    return {
                        "selected_option": sel,
                        "solution_text": {
                            "answer": final_answer,
                            "reasoning_steps": reasoning_steps
                        },
                        "audit_trail": audit
                    }
        
        # Original ReAct loop (fallback if refinement disabled or fails)
        print("\n[Executor] Falling back to standard ReAct loop...")
        audit.append("using_standard_react")
        history: List[str] = []
        for step in range(self.max_steps):
            obj = self._llm_action(problem, topic, options, history, exemplars, goals)
            if not obj or not isinstance(obj, dict):
                audit.append("LLM_parse_error_or_non_dict")
                break
            action_type = (obj.get("action_type") or "").upper()
            if action_type == "TOOL":
                tool = obj.get("tool","")
                args = obj.get("args",{})
                obs = self.tools.run(tool, args)
                if obs.get("ok"):
                    history.append(f"{tool}=>ok_keys={list(obs.get('data',{}).keys())}")
                else:
                    history.append(f"{tool}=>ERR:{obs.get('error')}")
                audit.append(f"tool:{tool}")
                continue
            elif action_type == "FINAL":
                final = obj.get("final", {}) if isinstance(obj, dict) else {}
                ans = final.get("answer", "")
                reason = final.get("reasoning_summary", "")

                try:
                    print(f"[Executor.solve] FINAL obj: {json.dumps(obj)}")
                except Exception:
                    print(f"[Executor.solve] FINAL obj (non-json): {str(obj)}")
                audit.append("LLM_emitted_FINAL")

                extracted = self._extract_answer_from_text(ans, options)
                if extracted:
                    audit.append("final_quick_extracted")
                    return {
                        "selected_option": extracted,
                        "solution_text": {"answer": str(ans), "reason": reason},
                        "audit_trail": audit,
                    }

                if verifier:
                    try:
                        verify_input = {
                            "problem_text": problem,
                            "topic": topic,
                            "parsed_constraints": {},
                            "candidate": {"final_answer": ans, "audit_trail": history + [reason]},
                            "options": options
                        }
                        vout = verifier.verify(verify_input)
                        audit.append(f"verifier:{vout.get('status')}")
                        if vout.get("status") == "PASS":
                            normalized = vout.get("normalized_answer") or ans
                            try:
                                selected_option = int(str(normalized).strip())
                                if 1 <= selected_option <= len(options):
                                    return {
                                        "selected_option": selected_option,
                                        "solution_text": {"answer": str(ans), "reason": reason},
                                        "audit_trail": audit,
                                    }
                            except Exception:
                                pass

                            if options:
                                chooser = self.tools.run("choice_selector", {
                                    "computed_result": str(normalized),
                                    "options": options
                                })
                                if chooser.get("ok") and chooser.get("data", {}).get("success"):
                                    sel_raw = chooser.get("data", {}).get("data", {}).get("selected_index")
                                    try:
                                        sel = int(sel_raw)
                                        return {
                                            "selected_option": sel,
                                            "solution_text": {"answer": str(ans), "reason": reason},
                                            "audit_trail": audit,
                                        }
                                    except Exception:
                                        audit.append(f"choice_selector_bad_index:{sel_raw}")
                    except Exception as e:
                        audit.append(f"verifier_exception:{e}")

                if options:
                    audit.append("final_extraction_failed_using_choice_selector")
                    chooser = self.tools.run("choice_selector", {
                        "computed_result": f"{ans} {reason}",
                        "options": options
                    })
                    if chooser.get("ok") and chooser.get("data", {}).get("success"):
                        sel_raw = chooser.get("data", {}).get("data", {}).get("selected_index")
                        try:
                            sel = int(sel_raw)
                            return {
                                "selected_option": sel,
                                "solution_text": {"answer": str(ans), "reason": reason},
                                "audit_trail": audit,
                            }
                        except Exception:
                            audit.append(f"choice_selector_bad_index:{sel_raw}")
                    else:
                        audit.append(f"choice_selector_failed:{chooser.get('error')}")

                audit.append("absolute_fallback_with_answer")
                return {
                    "selected_option": 1,
                    "solution_text": {"answer": str(ans), "reason": reason},
                    "audit_trail": audit,
                }
        
        det_idx = 1
        det_reason = "fallback_max_steps_exhausted"
        if history:
            chooser = self.tools.run("choice_selector", {
                "computed_result": history[-1],
                "options": options
            })
            if chooser.get("ok") and chooser.get("data", {}).get("success"):
                sel = chooser.get("data", {}).get("data", {}).get("selected_index")
                if sel:
                    det_idx = int(sel)
                    det_reason = "fallback_choice_selector_from_history"
        audit.append(det_reason)
        return {
            "selected_option": int(det_idx),
            "solution_text": det_reason,
            "audit_trail": audit
        }