"""Register the Arctic Embed model to Unity Catalog via MLflow."""
import argparse
import inspect
import os
import sys

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

# In Databricks serverless, __file__ is not defined. Use inspect to find
# this script's directory so we can import arctic_embed_model.py.
_here = os.path.dirname(os.path.abspath(
    inspect.currentframe().f_code.co_filename
))
sys.path.insert(0, _here)
from arctic_embed_model import ArcticEmbedModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    uc_model_name = f"{args.catalog}.{args.schema}.{args.model_name}"
    print(f"Registering model as: {uc_model_name}")

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/rob.bajra@databricks.com/{args.model_name}")

    model = ArcticEmbedModel()

    # Define signature manually — we cannot run inference at registration
    # time because the model requires a GPU.
    input_schema = Schema([
        ColSpec("string", "text"),
        ColSpec("string", "text_type"),
    ])
    output_schema = Schema([
        ColSpec("string", "embedding"),
    ])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    sample_input = pd.DataFrame({
        "text": ["What is machine learning?"],
        "text_type": ["query"],
    })

    with mlflow.start_run() as run:
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model,
            code_paths=[os.path.join(_here, "arctic_embed_model.py")],
            signature=signature,
            input_example=sample_input,
            pip_requirements=[
                "mlflow>=3.0",
                "pandas",
                "torch==2.5.1",
                "transformers==4.39.3",
                "accelerate>=0.25",
                "sentencepiece",
            ],
            registered_model_name=uc_model_name,
        )
        print(f"Model logged: {model_info.model_uri}")
        print(f"Run ID: {run.info.run_id}")

    # Get the latest version number
    from mlflow.tracking import MlflowClient
    client = MlflowClient(registry_uri="databricks-uc")
    versions = client.search_model_versions(
        filter_string=f"name='{uc_model_name}'",
    )
    version = max(int(v.version) for v in versions)
    print(f"Registered version: {version}")

    # Pass version to the next task via task values
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        from pyspark.dbutils import DBUtils
        dbutils = DBUtils(spark)
        dbutils.jobs.taskValues.set(key="model_version", value=str(version))
        print(f"Set task value model_version={version}")
    except Exception as e:
        print(f"Could not set task value (may not be running in a job): {e}")


if __name__ == "__main__":
    main()
