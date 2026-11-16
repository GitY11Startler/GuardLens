"""Loaded by pytest before any test module is imported."""

from guardlens.cli import use_project_hf_cache

# huggingface_hub reads HF_HOME once at import time, so this must run before test modules
# (which import `datasets` / `sentence_transformers`) are collected.
use_project_hf_cache()
