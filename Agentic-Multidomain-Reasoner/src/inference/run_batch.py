# src/inference/run_batch.py
"""
FIXED: Better solution extraction to populate CSV solution column
"""
import argparse
import pandas as pd
import os
import sys
import inspect
import hashlib
from src.inference.format_output_csv import normalize_dataframe, map_answer_to_option

def ensure_skill_db(train_csv: str, db_path: str):
    """Build skill DB if missing by invoking our build_index module."""
    if os.path.exists(db_path):
        return
    try:
        from src.skill_cache.build_index import build as build_fn
    except Exception as e:
        raise RuntimeError("build_index not importable: " + str(e))
    build_fn(train_csv, db_path)

def safe_construct(cls, *args, **kwargs):
    try:
        return cls(*args, **kwargs)
    except Exception:
        try:
            return cls()
        except Exception:
            return None

def _try_inject_exemplars_into_executor(executor, exemplars):
    """Attach exemplars non-invasively to executor/planner so planner can consume them."""
    try:
        if exemplars is None:
            return False
        if hasattr(executor, "planner"):
            try:
                setattr(executor.planner, "last_exemplars", exemplars)
                print("[run_batch] attached exemplars to executor.planner.last_exemplars")
                return True
            except Exception:
                pass
        try:
            setattr(executor, "last_exemplars", exemplars)
            print("[run_batch] attached exemplars to executor.last_exemplars")
            return True
        except Exception:
            pass
    except Exception:
        pass
    return False

def _call_executor_with_flex(executor, problem, topic, options, exemplars):
    """Try to call the executor using multiple candidate method names and argument orders."""
    def _safe_call(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:
            raise

    candidates = []
    for name in ("solve", "run_case", "run", "solve_case", "run_example", "process"):
        if hasattr(executor, name):
            candidates.append(getattr(executor, name))
    if not candidates:
        raise RuntimeError("No candidate executor method found on executor instance")

    last_exc = None
    for fn in candidates:
        try:
            sig = None
            try:
                sig = inspect.signature(fn)
            except Exception:
                sig = None
            if sig and "exemplars" in sig.parameters:
                try:
                    print(f"[run_batch] Attempting call {fn.__name__}(problem, topic, options, exemplars=...)")
                    return _safe_call(fn, problem, topic, options, exemplars=exemplars) or {}
                except TypeError as te:
                    last_exc = te
            if sig:
                params = sig.parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    try:
                        print(f"[run_batch] Attempting call {fn.__name__}(..., exemplars=...) via **kwargs support")
                        return _safe_call(fn, problem, topic, options, exemplars=exemplars) or {}
                    except TypeError as te:
                        last_exc = te
            try:
                print(f"[run_batch] Attempting positional call {fn.__name__}(problem, topic, options, exemplars)")
                return _safe_call(fn, problem, topic, options, exemplars) or {}
            except TypeError as te:
                last_exc = te
            try:
                print(f"[run_batch] Attempting positional call {fn.__name__}(problem, topic, options)")
                return _safe_call(fn, problem, topic, options) or {}
            except TypeError as te:
                last_exc = te
            try:
                print(f"[run_batch] Attempting positional call {fn.__name__}(problem, options, topic)")
                return _safe_call(fn, problem, options, topic) or {}
            except TypeError as te:
                last_exc = te
            try:
                print(f"[run_batch] Attempting positional call {fn.__name__}(problem, options)")
                return _safe_call(fn, problem, options) or {}
            except TypeError as te:
                last_exc = te
            try:
                print(f"[run_batch] Attempting dict call {fn.__name__}({'{...}'})")
                return _safe_call(fn, {"problem": problem, "topic": topic, "options": options, "exemplars": exemplars}) or {}
            except TypeError as te:
                last_exc = te
        except Exception as e:
            last_exc = e

    try:
        _try_inject_exemplars_into_executor(executor, exemplars)
        fn = candidates[0]
        try:
            print(f"[run_batch] Final fallback call {fn.__name__}(problem, topic, options) after attaching exemplars")
            return fn(problem, topic, options) or {}
        except Exception as e:
            last_exc = e
    except Exception:
        pass

    print(f"[run_batch] All attempts to call executor failed. Last exception: {last_exc}")
    return {"solution": None, "final_answer": None, "reasoning": None}

def _extract_solution_text(res: dict) -> str:
    """FIXED: Extract solution text from executor response, including reasoning steps."""
    if not isinstance(res, dict):
        return str(res) if res else ""
    
    # Handle refined solution with reasoning_steps
    if "solution_text" in res:
        sol_text = res["solution_text"]
        if isinstance(sol_text, dict):
            parts = []
            
            # Include reasoning steps if available
            if "reasoning_steps" in sol_text and sol_text["reasoning_steps"]:
                steps = sol_text["reasoning_steps"]
                if isinstance(steps, list):
                    parts.extend(steps)
            
            # Include answer/final answer
            for key in ["answer", "final_answer", "reason", "reasoning_summary", "reasoning"]:
                if key in sol_text and sol_text[key]:
                    parts.append(str(sol_text[key]))
            
            if parts:
                return " | ".join(parts)
            return str(sol_text)
        elif sol_text:
            return str(sol_text)

    # Handle direct solution object (from refinement)
    sol_candidates = []
    for key in ["reasoning_steps", "final_answer", "solution", "answer", "assistant_answer", "reasoning_summary", "reasoning"]:
        if key in res and res[key] is not None:
            val = res[key]
            # If it's a list (like reasoning_steps), join it
            if isinstance(val, list):
                sol_candidates.append(" | ".join(str(v) for v in val))
            else:
                sol_candidates.append(str(val))
    
    if sol_candidates:
        return " | ".join(sol_candidates)
    
    if "final" in res and isinstance(res["final"], dict):
        final = res["final"]
        parts = []
        for key in ["answer", "final_answer", "reasoning_summary"]:
            if key in final and final[key]:
                parts.append(str(final[key]))
        if parts:
            return " | ".join(parts)
    
    if "audit_trail" in res and res["audit_trail"]:
        return f"Audit: {', '.join(res['audit_trail'][-3:])}"
    
    return ""

def run_batch(test_csv: str, output_csv: str, cache_dir: str = None):
    df = pd.read_csv(test_csv)
    cache_dir = cache_dir or os.path.join(os.path.dirname(test_csv), "..", "cache")
    sqlite_path = os.path.join(cache_dir, "sqlite", "skills.db")
    train_csv_guess = os.path.join(os.path.dirname(test_csv), "train.csv")

    if not os.path.exists(sqlite_path):
        if os.path.exists(train_csv_guess):
            try:
                print("[run_batch] skills DB missing, attempting build from train CSV")
                ensure_skill_db(train_csv_guess, sqlite_path)
            except Exception as e:
                print("[run_batch] skill DB build failed:", e)
        else:
            print("[run_batch] skills DB missing and train CSV not found at", train_csv_guess)

    index_store = None
    try:
        from src.skill_cache.index_store import IndexStore
        if os.path.exists(sqlite_path):
            index_store = IndexStore(sqlite_path)
            print("[run_batch] IndexStore loaded")
    except Exception:
        index_store = None

    # Only showing the section that needs updating in run_batch.py
# Replace lines ~40-50 (the Executor construction section)

    try:
        from src.agent.executor import Executor
    except Exception as e:
        raise RuntimeError("Could not import Executor from src.agent.executor: " + str(e))

    # NEW: Import params for refinement config
    try:
        from src.agent.params import (
            ENABLE_SELF_REFINEMENT, 
            MAX_REFINEMENT_ITERATIONS, 
            ENABLE_COT_VERIFICATION
        )
    except Exception:
        ENABLE_SELF_REFINEMENT = True
        MAX_REFINEMENT_ITERATIONS = 2
        ENABLE_COT_VERIFICATION = True

    try:
        from src.agent.llm_client import OllamaClient
        llm = OllamaClient()
        executor = safe_construct(Executor, llm)
        if executor is None:
            executor = safe_construct(Executor)
        
        # NEW: Configure refinement settings
        if executor and hasattr(executor, 'enable_refinement'):
            executor.enable_refinement = ENABLE_SELF_REFINEMENT
            executor.max_refinement_iterations = MAX_REFINEMENT_ITERATIONS
            executor.enable_cot_verification = ENABLE_COT_VERIFICATION
            print(f"[run_batch] Refinement config: enabled={ENABLE_SELF_REFINEMENT}, "
                  f"max_iters={MAX_REFINEMENT_ITERATIONS}, cot_verify={ENABLE_COT_VERIFICATION}")
    except Exception:
        executor = safe_construct(Executor)

    # ... rest of run_batch.py stays the same

    try:
        from src.agent.llm_client import OllamaClient
        llm = OllamaClient()
        executor = safe_construct(Executor, llm)
        if executor is None:
            executor = safe_construct(Executor)
    except Exception:
        executor = safe_construct(Executor)

    outputs = []
    try:
        from src.skill_cache.sqlite_store import SQLiteStore
        compute_sha = SQLiteStore.compute_problem_sha
    except Exception:
        compute_sha = lambda s: hashlib.sha256((s or "").strip().lower().encode("utf-8")).hexdigest()

    for i, row in df.iterrows():
        print(f"\n[run_batch] Processing row {i+1}/{len(df)}")
        problem = str(row.get("problem_statement") or "")
        topic = str(row.get("topic") or "")
        options = []
        for j in range(1, 20):
            key = f"answer_option_{j}"
            if key in row and pd.notna(row[key]):
                options.append(str(row[key]))

        exemplar_list = None
        if index_store:
            problem_sha = compute_sha(problem)
            hit = index_store.exact_lookup(problem_sha)
            if hit:
                exemplar_list = [hit]
            else:
                try:
                    # CRITICAL: retrieve_similar returns [{"score": float, "record": dict}, ...]
                    sims = index_store.retrieve_similar(problem_text=problem, topic=topic, top_k=3, min_score=0.0)
                    recs = []
                    for s in sims:
                        # Extract score and record from the dict
                        if isinstance(s, dict):
                            rec = s.get("record", {})
                            sim_score = s.get("score", 0.0)  # KEY FIX: It's "score" not "similarity"
                        else:
                            rec = s
                            sim_score = 0.0

                        if rec and isinstance(rec, dict):
                            # Attach similarity to the record
                            rec["similarity"] = sim_score
                            recs.append(rec)

                    if recs:
                        exemplar_list = recs
                        # Display proper similarity
                        top_sim = recs[0].get("similarity", 0.0)
                        print(f"[run_batch] Retrieved {len(exemplar_list)} exemplars, top_sim={top_sim:.4f}")
                except Exception as e:
                    print(f"[run_batch] Retrieval failed: {e}")
                    import traceback
                    traceback.print_exc()
                    exemplar_list = None

        # Attach exemplars to executor
        try:
            if exemplar_list:
                _try_inject_exemplars_into_executor(executor, exemplar_list)
        except Exception:
            pass

        # call executor
        try:
            res = _call_executor_with_flex(executor, problem, topic, options, exemplar_list)
        except Exception as e:
            print(f"[run_batch] Executor failed on row {i}: {e}")
            res = {"solution_text": None, "selected_option": None}

        out_row = dict(row)
        solution_text = _extract_solution_text(res)
        out_row["solution"] = solution_text if solution_text else "No solution generated"

        selected_option = None
        if isinstance(res, dict):
            selected_option = res.get("selected_option")

        if selected_option is None:
            mapped = map_answer_to_option(solution_text, options)
            selected_option = mapped

        out_row["correct_option_number_pred"] = selected_option if selected_option else 1
        print(f"[run_batch] Row {i+1}: selected_option={out_row['correct_option_number_pred']}, solution_preview={solution_text[:100]}")
        outputs.append(out_row)

    out_df = pd.DataFrame(outputs)
    out_df = normalize_dataframe(out_df)
    out_df.to_csv(output_csv, index=False)
    print(f"\n[run_batch] Wrote output to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("test_csv")
    p.add_argument("output_csv")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()
    run_batch(args.test_csv, args.output_csv, cache_dir=args.cache_dir)
