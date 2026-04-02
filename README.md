# serve-recommendations

A Databricks Model Serving endpoint that serves item recommendations using a custom MLflow pyfunc model. The model is instantiated with a dictionary of category-to-item mappings and uses a three-tier heuristic to match inputs to recommendations.

## How It Works

The model accepts a `category` string and returns ranked recommendations using:

1. **Exact match** — direct category lookup (e.g., `"electronics"`)
2. **Partial match** — substring matching (e.g., `"book"` matches `"books"`)
3. **Fallback** — default recommendations when no match is found

Built-in categories: `electronics`, `books`, `clothing`, `home`, `sports`

## Project Structure

```
model-serving-example/
├── databricks.yml                  # Databricks Asset Bundle config
├── resources/
│   ├── deploy_model_job.yml        # Serverless job: register model in Unity Catalog
│   └── serving_endpoint.yml        # Model serving endpoint (native DAB resource)
└── src/
    ├── recommendation_model.py     # PyFunc model class + recommendation data
    └── register_model.py           # Logs model to MLflow, registers in Unity Catalog
```

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.200+
- A Databricks workspace with Unity Catalog enabled
- A CLI profile configured (default profile name: `DEV`)

## Setup

1. Create the Unity Catalog schema (if it doesn't exist):

   ```sql
   CREATE SCHEMA IF NOT EXISTS main.model_serving;
   ```

2. Validate the bundle:

   ```bash
   cd model-serving-example
   databricks bundle validate -t dev
   ```

3. Register the model (run the job on serverless compute):

   ```bash
   databricks bundle run register_recommendation_model -t dev
   ```

   This logs the pyfunc model to MLflow and registers it in `main.model_serving.recommendation_model`.

4. Deploy the bundle (creates the serving endpoint):

   ```bash
   databricks bundle deploy -t dev
   ```

   The serving endpoint is defined as a native DAB resource and will be created automatically with scale-to-zero enabled.

5. Wait for the endpoint to become ready (~5-10 minutes).

## Testing the Endpoint

```bash
curl -X POST \
  https://<workspace-url>/serving-endpoints/dev-recommendation-endpoint/invocations \
  -H "Authorization: Bearer $(databricks auth token --profile DEV -p)" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [
      {"category": "electronics"},
      {"category": "book"},
      {"category": "gardening"}
    ]
  }'
```

**Response:**

```json
{
  "predictions": [
    {
      "recommended_items": "[{\"item\": \"Wireless Headphones\", \"score\": 0.95}, ...]",
      "match_type": "exact"
    },
    {
      "recommended_items": "[{\"item\": \"Python Crash Course\", \"score\": 0.93}, ...]",
      "match_type": "partial"
    },
    {
      "recommended_items": "[{\"item\": \"Gift Card\", \"score\": 0.7}, ...]",
      "match_type": "default"
    }
  ]
}
```

## Customizing Recommendations

Edit the `DEFAULT_RECOMMENDATIONS` dictionary in `src/recommendation_model.py` to change categories and items. Each category maps to a list of `{"item": str, "score": float}` entries. Then re-register and redeploy:

```bash
databricks bundle run register_recommendation_model -t dev
databricks bundle deploy -t dev
```

## Targets

| Target | Endpoint Name | Profile |
|--------|--------------|---------|
| `dev` | `dev-recommendation-endpoint` | `DEV` |
| `prod` | `recommendation-endpoint` | `DEV` |

## Cleanup

```bash
databricks bundle destroy -t dev
```
