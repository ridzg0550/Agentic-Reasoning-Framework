"""Centralized configuration for the agentic reasoning system."""
# LLM Configuration
OLLAMA_API_BASE_URL = "https://993440102f2e.ngrok-free.app/api/generate"
LLM_MODEL_NAME = "llama2:13b"

# Agent Configuration
MAX_EXECUTION_STEPS = 7
MAX_PLANNER_GOALS = 5

# NEW: Self-Refinement Configuration
ENABLE_SELF_REFINEMENT = True
MAX_REFINEMENT_ITERATIONS = 2  # Number of critique-improve cycles

# NEW: Chain-of-Thought Verification
ENABLE_COT_VERIFICATION = True  # Verify each reasoning step

# Skill Cache Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_SIMILARITY_THRESHOLD = 0.80  # lowered to admit near-duplicates
CACHE_TOP_K = 5
SIMHASH_BIT_DIFFERENCE = 10

# Tool Configuration
PYTHON_TOOL_TIMEOUT = 2  # seconds
PYTHON_TOOL_MAX_MEMORY = 128  # MB
RETRIEVER_TOP_K_EXEMPLARS = 3