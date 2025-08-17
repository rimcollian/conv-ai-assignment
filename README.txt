
Assignment 2 – Comparative Financial QA System: RAG vs Fine-Tuning
Group 88

FILES
-----
- Assignment2_RAG_vs_FT_RIL_Group88.ipynb : End-to-end notebook with RAG + Fine-tuning scaffolds, advanced techniques, evaluation stubs.
- app.py : Streamlit UI that switches between RAG and Fine-Tuned modes, with re-ranking + guardrails.
- ril_fin_qa_pairs.csv : ~56 Q/A pairs derived from RIL Integrated Annual Reports FY2023-24 & FY2022-23.
- ril_fin_qa_finetune.jsonl : Instruction-tuning dataset created from the Q/A pairs.
- Assignment2_Report_Group88.pdf : Report scaffold with placeholders for 3 screenshots + summary table.
- README.txt : This help file.

HOW TO RUN (LOCAL)
------------------
1) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
2) pip install -U transformers sentence-transformers faiss-cpu rank_bm25 scikit-learn streamlit torch accelerate peft bitsandbytes datasets reportlab
3) streamlit run app.py
4) Open the notebook and run sections 1→5. For fine-tuning, enable a GPU (Colab/Kaggle).

ADVANCED TECHNIQUES INCLUDED
----------------------------
- Hybrid retrieval (BM25 + MiniLM) with optional cross-encoder re-ranking.
- Fine-tuning with PEFT LoRA; Mixture-of-Experts scaffold (multi-LoRA + learned gating).
- Guardrails: input (scope/unsafe) and output (numerical cross-check).

WORKAROUNDS / ACTION NEEDED
---------------------------
- This environment cannot download models; please run notebook/app on your machine or Colab with internet.
- Take 3 screenshots of the UI showing: query, answer, confidence, response time, method used. Paste into the PDF (or keep as separate images if allowed).
- If hosting is required, deploy app.py to Hugging Face Spaces or Streamlit Community Cloud and paste the public URL into the PDF.

