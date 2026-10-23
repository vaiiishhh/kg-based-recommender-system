# KG_Recommender

This project is a Knowledge Graph (KG) based recommender system that uses a Large Language Model (LLM) to perform collaborative filtering and predictions based on user-item interaction paths.

## Project Structure

- `src/`: Contains all the Python source code for building the KG, extracting paths, generating prompts, and making predictions.
- `data/`: Contains the Knowledge Graph data (`kg_data/`), ground truth datasets, and prediction outputs.
- `docs/`: Documentation, including a LaTeX report summarizing the methodology and findings.
- `tests/`: Directory for future tests.

## Key Components

The codebase follows a modular pipeline structure:

1. **Building the KG** (Steps 1 to 6): Found in `src/kg_entity_context.py` and `src/kg_run_demo.py` which load the entity mappings, features, categories, and brands to index them.
2. **Path Extraction**: `src/kg_path_extractor.py` and `src/kg_path_to_json.py` handle finding relationships (paths) between users and items using search algorithms.
3. **LLM & Prompt Builder**: `src/llm_prompt_builder.py` and `src/prompt_variants.py` construct reasoning prompts using the extracted paths and entity data.
4. **Prediction**: `src/predict.py` executes the API call to the LLM and processes output scores. `src/run_final_pipeline.py` provides an end-to-end execution.

## Setup & Requirements

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **LLM Server Setup**:
   This project uses a server to run an LLM locally. To run it on your own system:
   - Ensure your local LLM server is up and running (e.g. Ollama, vLLM, or any compatible OpenAI drop-in replacement).
   - Change the API key and Base URL to match your server configuration. You can update these settings directly inside `src/predict.py`.

## How to Run

1. Make sure your data is in the `data/kg_data/` directory.
2. Ensure your local LLM server is active and the API configurations are updated.
3. You can run the end-to-end pipeline (KG path extraction -> LLM predictions) using:
   ```bash
   python src/run_final_pipeline.py
   ```
4. To evaluate the prediction results using NDCG metrics, run:
   ```bash
   python src/evaluate_ndcg.py
   ```
