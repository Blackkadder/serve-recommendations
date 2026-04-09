"""Snowflake Arctic Embed L v2.0 — custom MLflow pyfunc for GPU model serving."""
import json
import os

import mlflow
import numpy as np
import pandas as pd


class ArcticEmbedModel(mlflow.pyfunc.PythonModel):
    """Wraps Snowflake/snowflake-arctic-embed-l-v2.0 as an MLflow pyfunc.

    Produces 1024-dimensional L2-normalised embeddings. Accepts an optional
    ``text_type`` column ("query" or "document") — queries are prefixed with
    ``"query: "`` for optimal retrieval quality per the model card.
    """

    MODEL_ID = "Snowflake/snowflake-arctic-embed-l-v2.0"

    def load_context(self, context):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = (
            AutoModel.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float16,
                add_pooling_layer=False,
            )
            .to("cuda")
            .eval()
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["text"].tolist()

        # Apply "query: " prefix when text_type == "query"
        if "text_type" in model_input.columns:
            text_types = model_input["text_type"].fillna("document").tolist()
        else:
            text_types = ["document"] * len(texts)

        prefixed = [
            f"query: {t}" if tt == "query" else t
            for t, tt in zip(texts, text_types)
        ]

        embeddings = self._embed_batch(prefixed)

        return pd.DataFrame({
            "embedding": [json.dumps(vec.tolist()) for vec in embeddings],
        })

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Tokenize, forward-pass, CLS-pool, and L2-normalise."""
        import torch
        import torch.nn.functional as F

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**encoded)

        # CLS pooling — first token of last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0]
        normalised = F.normalize(cls_embeddings, p=2, dim=1)

        return normalised.cpu().float().numpy()
