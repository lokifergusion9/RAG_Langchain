# RAG with LangChain

Hands-on experiments in retrieval-augmented generation (RAG): load PDFs, chunk and embed them, retrieve relevant passages, answer with an LLM, and then **measure** answer quality instead of eyeballing it.

## What's inside

| File | What it does |
|---|---|
| [`Rag/rag.ipynb`](Rag/rag.ipynb) | Basic pipeline: `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000/200 and 800/200 chunk/overlap) → OpenAI embeddings → in-memory (DocArray) and Chroma vector stores → prompt + `ChatOpenAI`. Includes page and line-number metadata so answers can cite their source, and a multi-PDF index. |
| [`Rag/perplexity.ipynb`](Rag/perplexity.ipynb) | Swaps the answer model between OpenAI and Perplexity (OpenAI-compatible API) and evaluates each variant with **Giskard** on an auto-generated test set. |
| [`Rag/test-set.json`](Rag/test-set.json) | The Giskard-generated question / reference-answer set built from the *Attention Is All You Need* paper. |
| `Rag/*.pdf`, `Rag/speech.txt` | Small sample corpus (a paper, personal study notes, a text file). |

## Run it

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file next to the notebooks (never commit it):

```
openai_api_key=...
perplexity_api_key=...
```

Then open the notebooks in Jupyter. They load the PDFs by relative path, so launch Jupyter from inside `Rag/`.

## Notes

- This is an experiments repo, not a library. The notebooks are meant to be read top to bottom.
- The Perplexity model name used (`llama-3.1-sonar-huge-128k-online`) has since been retired; substitute a current Perplexity model to rerun that notebook.
- Only the top retrieved chunk is passed to the model in the evaluation functions, which is a deliberate simple baseline to improve on (more chunks, reranking, hybrid search).
