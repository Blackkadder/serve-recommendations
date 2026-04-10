# GPU Serving — Multi-Instance with Load Balancing

Demonstrates GPU co-location by serving **two instances** of
[Snowflake/snowflake-arctic-embed-l-v2.0](https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0)
behind a single Databricks Model Serving endpoint on a GPU_SMALL (T4 16GB).

This pattern reduces cost by sharing one GPU across multiple model instances
(or different models) instead of provisioning a dedicated endpoint for each.

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Databricks Model Serving (GPU_SMALL)   │
│                                                 │
│  ┌───────────────────────────────────────────┐  │
│  │         MultiEmbedModel (pyfunc)          │  │
│  │                                           │  │
│  │  model_name ──► route ──► instance        │  │
│  │       or                                  │  │
│  │  (absent)  ──► load balancer ──► instance │  │
│  │                                           │  │
│  │  ┌─────────────┐   ┌─────────────┐       │  │
│  │  │  model-a    │   │  model-b    │       │  │
│  │  │ Arctic L v2 │   │ Arctic L v2 │       │  │
│  │  │  (fp16)     │   │  (fp16)     │       │  │
│  │  └─────────────┘   └─────────────┘       │  │
│  │         CUDA device (shared)              │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Request Format

```json
{
    "dataframe_records": [
        {"text": "What is ML?", "text_type": "query", "model_name": "model-a"},
        {"text": "ML overview.", "text_type": "document", "model_name": "model-b"},
        {"text": "No preference.", "text_type": "query"}
    ]
}
```

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `text` | string | Yes | Text to embed (max 8192 tokens) |
| `text_type` | string | No | `"query"` or `"document"` (default: `"document"`) |
| `model_name` | string | No | `"model-a"` or `"model-b"`. Omit for load-balanced routing. |

## Load Balancing Strategies

Set `BALANCE_STRATEGY` on the `MultiEmbedModel` class in `src/multi_embed_model.py`.

| Strategy | Behaviour | Best For |
|----------|-----------|----------|
| `round_robin` (default) | Alternates between instances per unrouted row | Even distribution |
| `batch_split` | Splits unrouted rows 50/50 across instances | Large batches — distributes GPU memory pressure |
| `random` | Random assignment per row | Unpredictable traffic patterns |

## VRAM Budget

| Component | Size |
|-----------|------|
| Arctic Embed L v2.0 (fp16) × 2 | ~2.2 GB |
| Tokenizer buffers (peak) | ~1–2 GB |
| **Total** | **~3–4 GB** |
| T4 capacity | 16 GB |
| **Headroom** | **~12 GB** |

## Adapting for Different Models

To co-locate different models (e.g., `all-mpnet-base-v2` + `multilingual-e5-base`):

1. Edit `MODEL_INSTANCES` in `src/multi_embed_model.py` to use different HuggingFace IDs
2. If the new model uses **mean pooling** instead of CLS pooling, update `_embed_batch`
3. If the new model has different prefix conventions, update the prefixing logic in `predict`
4. Verify combined VRAM fits on GPU_SMALL (T4 16GB) — upgrade to GPU_MEDIUM (A10G 24GB) if needed

## Deployment

```bash
# 1. Deploy the registration job (without endpoint first)
mv resources/serving_endpoint.yml serving_endpoint.yml.bak
databricks bundle deploy -t dev

# 2. Run registration (logs model to Unity Catalog)
databricks bundle run register_multi_embed_model -t dev

# 3. Deploy the endpoint
mv serving_endpoint.yml.bak resources/serving_endpoint.yml
databricks bundle deploy -t dev
# Note: GPU endpoint takes 10-20 min to start (pip install + model download)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.5.1 | CUDA 12.4 compatible |
| transformers | 4.39.3 | Avoids torchvision import issues |
| accelerate | >=0.25 | Multi-GPU support |
| sentencepiece | latest | Tokenizer dependency |
| mlflow | >=3.0 | Pyfunc serving runtime |
| pandas | latest | DataFrame I/O |
