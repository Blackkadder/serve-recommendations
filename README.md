# serve-recommendations

A Databricks Model Serving endpoint that serves item recommendations using a custom MLflow pyfunc model. The model uses a three-tier heuristic (exact match, partial match, fallback) and streams prediction telemetry to a Unity Catalog table via Zerobus Ingest.

## Folders

- **`model-serving-example/`** -- A custom MLflow pyfunc model served via Databricks Model Serving (see details below).
- **`gpu-serving/`** -- An example of using Databricks to host Hugging Face models as model serving endpoints.

## How It Works

The model accepts a `category` string and returns ranked recommendations:

1. **Exact match** -- direct category lookup (e.g., `"electronics"`)
2. **Partial match** -- substring matching (e.g., `"book"` matches `"books"`)
3. **Fallback** -- default recommendations when no match is found

Each prediction is logged to `main.model_serving.recommendation_logs` via Zerobus Ingest with the OTel trace ID for correlation with MLflow traces.

Built-in categories: `electronics`, `books`, `clothing`, `home`, `sports`

## Project Structure

```
model-serving-example/
├── databricks.yml                        # DAB bundle config (variables, targets)
├── resources/
│   ├── deploy_model_job.yml              # Job: register model in Unity Catalog
│   ├── serving_endpoint.yml              # Model serving endpoint config
│   └── monitoring_dashboard.yml          # AI/BI monitoring dashboard
└── src/
    ├── recommendation_model.py           # PyFunc model + Zerobus telemetry
    ├── register_model.py                 # Logs & registers model in UC
    ├── create_logs_table.sql             # DDL for telemetry table + grants
    └── model_serving_monitor.lvdash.json # Dashboard definition
```

## Databricks Asset Bundles (DABs)

This project uses [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html) to manage all infrastructure as code. DABs provide:

- **Declarative resource definitions** -- jobs, endpoints, dashboards defined in YAML
- **Multi-environment targeting** -- `dev` and `prod` targets with variable overrides
- **Terraform-backed deployment** -- resources are provisioned and tracked via state
- **Bundle variables** -- parameterize catalog, schema, endpoint names, warehouse IDs across environments

### Key Concepts

| Concept | Description |
|---------|-------------|
| `databricks.yml` | Root config declaring bundle name, variables, targets, and resource includes |
| `resources/*.yml` | Resource definitions (jobs, endpoints, dashboards) auto-included via `include` |
| `targets` | Environment-specific overrides (workspace profile, variable values, dev/prod mode) |
| `variables` | Parameterized values shared across resources, overridable per target |
| `bundle deploy` | Uploads files and provisions/updates all declared resources |
| `bundle run` | Triggers a specific job or pipeline defined in the bundle |
| `bundle destroy` | Tears down all provisioned resources |

### Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `catalog` | `main` | Unity Catalog catalog |
| `schema` | `model_serving` | UC schema for model and tables |
| `model_name` | `recommendation_model` | Registered model name |
| `endpoint_name` | `recommendation-endpoint` | Serving endpoint name |
| `warehouse_id` | `4312e046b55e5caf` | SQL warehouse for dashboard |

## Prerequisites

- [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html) v0.200+
- A Databricks workspace with Unity Catalog enabled
- A CLI profile configured (default: `DEV`)
- A service principal for Zerobus Ingest with `MODIFY`/`SELECT` on the telemetry table
- A `zerobus` secret scope with: `server_endpoint`, `workspace_url`, `client_id`, `client_secret`

## Setup

1. Create the Unity Catalog schema and telemetry table:

   ```sql
   CREATE SCHEMA IF NOT EXISTS main.model_serving;
   ```

   Then run `src/create_logs_table.sql` to create the telemetry table and grant the service principal access.

2. Create the secret scope (one-time):

   ```bash
   databricks secrets create-scope zerobus
   databricks secrets put-secret zerobus server_endpoint --string-value "<workspace-id>.zerobus.<region>.cloud.databricks.com"
   databricks secrets put-secret zerobus workspace_url --string-value "https://<workspace-url>"
   databricks secrets put-secret zerobus client_id --string-value "<sp-client-id>"
   databricks secrets put-secret zerobus client_secret --string-value "<sp-client-secret>"
   ```

3. Validate, register, and deploy:

   ```bash
   cd model-serving-example
   databricks bundle validate -t dev
   databricks bundle deploy -t dev
   databricks bundle run register_recommendation_model -t dev
   ```

4. Update `entity_version` in `resources/serving_endpoint.yml` to the newly registered version, then redeploy:

   ```bash
   databricks bundle deploy -t dev
   ```

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

Verify telemetry is flowing:

```sql
SELECT * FROM main.model_serving.recommendation_logs ORDER BY event_timestamp DESC LIMIT 10;
```

## CI/CD Best Practices

### Branch Strategy

Use a trunk-based workflow with DABs targets:

```
feature/* ──> develop (dev target) ──> main (prod target)
```

- **`develop`** -- deploys to dev workspace with `mode: development` (resource name prefixes, shorter retention)
- **`main`** -- deploys to prod workspace with `mode: production` (no prefixes, full permissions)

### Pipeline Stages

A typical CI/CD pipeline for DABs:

```
┌───────────┐    ┌──────────┐    ┌───────────┐    ┌────────────┐
│  Validate  │───>│  Deploy   │───>│  Register  │───>│  Deploy    │
│  (bundle   │    │  (dev)    │    │  Model     │    │  (prod)    │
│  validate) │    │           │    │  (bundle   │    │            │
│            │    │           │    │   run)     │    │            │
└───────────┘    └──────────┘    └───────────┘    └────────────┘
```

**1. Validate** (on every PR):
```bash
databricks bundle validate -t dev
databricks bundle validate -t prod
```

**2. Deploy to dev** (on merge to `develop`):
```bash
databricks bundle deploy -t dev
databricks bundle run register_recommendation_model -t dev
```

**3. Deploy to prod** (on merge to `main`):
```bash
databricks bundle deploy -t prod
databricks bundle run register_recommendation_model -t prod
```

### GitHub Actions Example

```yaml
name: Deploy
on:
  push:
    branches: [develop, main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Databricks CLI
        run: curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sh

      - name: Set target
        id: target
        run: |
          if [ "${{ github.ref_name }}" = "main" ]; then
            echo "target=prod" >> $GITHUB_OUTPUT
          else
            echo "target=dev" >> $GITHUB_OUTPUT
          fi

      - name: Validate
        run: databricks bundle validate -t ${{ steps.target.outputs.target }}
        working-directory: model-serving-example

      - name: Deploy
        run: databricks bundle deploy -t ${{ steps.target.outputs.target }}
        working-directory: model-serving-example

      - name: Register Model
        run: databricks bundle run register_recommendation_model -t ${{ steps.target.outputs.target }}
        working-directory: model-serving-example
```

### Secrets Management

- **Never commit credentials** -- use Databricks secret scopes referenced via `{{secrets/scope/key}}` in endpoint environment variables
- **Use service principals** for CI/CD authentication (`DATABRICKS_HOST` + `DATABRICKS_TOKEN` or OAuth)
- **Separate secret scopes per environment** if credentials differ between dev and prod

### Model Versioning

The current setup requires manually updating `entity_version` in `serving_endpoint.yml` after model registration. To automate:

1. The registration job already sets a task value (`model_version`)
2. A downstream task can update the endpoint via the API
3. Alternatively, use `databricks bundle deploy` with variable overrides:
   ```bash
   databricks bundle deploy -t dev --var="model_version=9"
   ```

## Targets

| Target | Endpoint Name | Mode | Profile |
|--------|--------------|------|---------|
| `dev` | `dev-recommendation-endpoint` | development | `DEV` |
| `prod` | `recommendation-endpoint` | production | `DEV` |

## Monitoring

The bundle includes an AI/BI dashboard (`Model Serving Monitor`) that tracks:

- **KPIs** -- total requests, avg latency, error rate
- **Latency percentiles** -- P50, P95, P99
- **Trends** -- request volume and latency over time
- **Distributions** -- latency histogram, status codes, match types
- **Recent requests** -- detailed request log with IDs and latency

Data source: `main.rob_test.model_serving_payload` (inference table)

## Customizing Recommendations

Edit `DEFAULT_RECOMMENDATIONS` in `src/recommendation_model.py`, then re-register and redeploy:

```bash
databricks bundle run register_recommendation_model -t dev
# Update entity_version in serving_endpoint.yml
databricks bundle deploy -t dev
```

## Cleanup

```bash
databricks bundle destroy -t dev
```
