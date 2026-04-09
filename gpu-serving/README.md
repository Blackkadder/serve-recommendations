# GPU Serving — Snowflake Arctic Embed L v2.0

Deploy [Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) on Databricks Model Serving with GPU compute using a custom MLflow pyfunc.

## Model

| Property | Value |
|----------|-------|
| Model | [snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0) |
| Type | Multilingual text embedding (74 languages) |
| Parameters | 568M |
| Output | 1024-dimensional L2-normalised vectors |
| Context | 8192 tokens |
| License | Apache 2.0 (open, no gated access) |
| GPU | GPU_SMALL (T4, 16GB) — model is ~1.1GB in FP16 |

## How It Works

The model is wrapped in a custom MLflow pyfunc (`ArcticEmbedModel`) that:

1. Loads the model and tokenizer onto GPU in `load_context` (runs once at container startup)
2. Accepts text input with an optional `text_type` column (`"query"` or `"document"`)
3. Prepends `"query: "` to query texts for optimal retrieval quality (per model card)
4. Batches all inputs through the model, applies CLS pooling and L2 normalisation
5. Returns 1024-dim embedding vectors as JSON arrays

## Project Structure

```
gpu-serving/
├── databricks.yml                # DAB bundle config (variables, targets)
├── resources/
│   ├── deploy_model_job.yml      # Job: register model in Unity Catalog
│   └── serving_endpoint.yml      # Model serving endpoint config (GPU_SMALL)
└── src/
    ├── arctic_embed_model.py     # Custom pyfunc model class
    └── register_model.py         # Logs & registers model in UC
```

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.200+
- A Databricks workspace with Unity Catalog enabled
- A CLI profile configured (default: `DEV`)

No HuggingFace token is needed — the model is fully open under Apache 2.0.

## Deployment

### 1. Validate the bundle

```bash
cd gpu-serving
databricks bundle validate -t dev
```

### 2. Deploy the job first (without endpoint)

The serving endpoint requires the model to be registered in Unity Catalog first. Temporarily move the endpoint config out, deploy the job, register the model, then restore.

```bash
# Move endpoint out temporarily
mv resources/serving_endpoint.yml serving_endpoint.yml.bak

# Deploy job only
databricks bundle deploy -t dev

# Run registration job
databricks bundle run register_arctic_embed_model -t dev
```

The registration job runs on serverless CPU — it logs the pyfunc class and dependency list to MLflow without downloading the model weights. The GPU dependencies (torch, transformers, accelerate) are installed at serving container startup time.

### 3. Deploy the endpoint

```bash
# Restore endpoint config
mv serving_endpoint.yml.bak resources/serving_endpoint.yml

# Update entity_version in resources/serving_endpoint.yml if needed
# (default is "1" which matches the first registration)

# Deploy with endpoint
databricks bundle deploy -t dev
```

The endpoint takes **10-20 minutes** to become ready — the container installs PyTorch, downloads model weights (~1.1GB), and loads the model to GPU. The `databricks bundle deploy` command may time out waiting; the endpoint will continue provisioning in the background.

Check status:

```bash
databricks serving-endpoints get dev_rob_bajra_dev-arctic-embed-endpoint --profile DEV
```

### 4. Test the endpoint

```bash
curl -X POST \
  https://<workspace-url>/serving-endpoints/dev-arctic-embed-endpoint/invocations \
  -H "Authorization: Bearer $(databricks auth token --profile DEV -p)" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"text": "What is machine learning?", "text_type": "query"},
      {"text": "Machine learning is a subset of AI.", "text_type": "document"}
    ]
  }'
```

Each record returns an `embedding` field containing a JSON array of 1024 floats.

## Input / Output

### Input

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `text` | string | Yes | The text to embed |
| `text_type` | string | No | `"query"` or `"document"` (default: `"document"`). Queries get a `"query: "` prefix for optimal retrieval. |

### Output

| Column | Type | Description |
|--------|------|-------------|
| `embedding` | string | JSON array of 1024 float values (L2-normalised) |

## Targets

| Target | Endpoint Name | Mode | Profile |
|--------|--------------|------|---------|
| `dev` | `dev-arctic-embed-endpoint` | development | `DEV` |
| `prod` | `arctic-embed-endpoint` | production | `DEV` |

## Key Design Decisions

- **`load_context` not lazy init**: The model loads once at container startup. Lazy loading in `predict` would cause request timeouts (120s limit) for a model that takes minutes to download.
- **Batch inference**: All input rows are processed in a single forward pass for GPU efficiency, unlike the row-by-row iteration in the `model-serving-example`.
- **GPU_SMALL (T4)**: The 568M parameter model is only ~1.1GB in FP16 — no need for the more expensive GPU_MEDIUM (A10G).
- **Scale-to-zero enabled**: Saves cost when idle, but cold starts take 10-20 minutes. For production with latency requirements, set `scale_to_zero_enabled: false`.
- **Manual signature**: The registration script defines the MLflow signature manually because it runs on CPU (no GPU to run inference for signature inference).

## Cleanup

```bash
databricks bundle destroy -t dev
```
