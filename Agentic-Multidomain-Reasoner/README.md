ETHOS: Self-Refinement Agentic Framework

A novel agentic AI system for autonomous multi-domain problem solving across spatial reasoning, logic puzzles, mathematical reasoning, and riddles.



🏗️ Architecture Overview


System Pipeline

Problem + Options → Skill Cache → Planner → Executor → Verifier → Answer
                         ↓           ↓          ↑
                    Exemplars    Goals    Self-Refinement Loop
                                            (Critique → Refine)

                                            
Core Components


1.Skill Cache (Long-Term Memory)

-Dual-mode retrieval: SHA-256 exact matching + FAISS semantic search

-SQLite for metadata storage

-384-dim embeddings via SentenceTransformer

-O(1) exact lookup, efficient k-NN for similar problems



2.Planner (Goal Decomposition)

-Generates 3-5 actionable sub-goals

-Leverages retrieved exemplars for few-shot learning

-Structured JSON output with schema validation



3.Executor (Self-Refinement Engine)

-Chain-of-Thought (CoT) reasoning for initial solutions

-Critique Module identifies logical flaws and errors

-Refinement Loop iteratively improves solutions (max 2 iterations)

-Falls back to ReAct loop if refinement disabled



4.Tool Dispatcher (Computational Backend)

-Python Tool: Sandboxed execution with AST validation

-SymPy Tool: Symbolic mathematics (equations, simplification)

-Z3 Tool: SMT constraint solving for logic problems

-Choice Selector: Fuzzy matching for answer normalization



5.Verifier (Answer Validation)

-Domain-specific logic (riddles, position puzzles)

-Fuzzy string matching (threshold ≥ 0.70)

-Answer normalization to option indices



6.Context Manager (Memory Optimization)

-Hierarchical compression: Header + Variables + Recent Steps




🔄 Self-Refinement Workflow     
┌─────────────────┐
│  CoT Reasoning  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      No Flaws
│    Critique     │─────────────────┐
└────────┬────────┘                 │
         │ Found Flaws              │
         ▼                          ▼
┌─────────────────┐          ┌──────────┐
│ Refine Solution │          │ Converged│
└────────┬────────┘          └──────────┘
         │
         └─────────────┐
                       ▼
              (Repeat max 2 iterations)



🚀 Key Features


Self-Correction: Iterative critique-improve loop for error recovery

Hybrid Retrieval: 35.3% exact hit rate + semantic fallback

Multi-Tool Support: Python, SymPy, Z3 with robust error handling

Domain-Aware: Specialized logic for different problem types

Efficient Context: Compressed history within token limits

Interpretability: Full audit trails for debugging


# Core Dependencies

python >= 3.8

torch >= 2.0.0

transformers >= 4.30.0

sentence-transformers >= 2.2.0

faiss-cpu >= 1.7.0  

sympy >= 1.8

z3-solver >= 4.8

pandas

numpy

sqlite3



🛠️ Installation

For Google Colab (Recommended)

1.Download the REpository

-download the repository as a ZIP file

-save it locally


2.Open Google Collab

-upload the ZIP file

-paste code block from https://colab.research.google.com/drive/1ILqMjDTLsLlCwppsPy2KfYr1LuaetP-E?usp=sharing


3.Start Running

-click on upload and upload the ZIP file

-then click on cancel upload


4.Download results

-download the output CSV
