"""Multi-instance Snowflake Arctic Embed L v2.0 — GPU co-location with load balancing."""
import json
import random

import mlflow
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model instances to load.  Each key is a logical name exposed to callers.
# Swap model IDs here to co-locate different HuggingFace models.
# ---------------------------------------------------------------------------
MODEL_INSTANCES = {
    "model-a": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "model-b": "Snowflake/snowflake-arctic-embed-l-v2.0",
}

VALID_STRATEGIES = ("round_robin", "batch_split", "random")


class MultiEmbedModel(mlflow.pyfunc.PythonModel):
    """Serves multiple model instances behind a single GPU endpoint.

    Two copies of Arctic Embed L v2.0 are loaded onto the same GPU.
    Callers may pin a request to a specific instance via the ``model_name``
    column, or omit it to let the built-in load balancer choose.

    Load-balancing strategy is set via the ``BALANCE_STRATEGY`` class
    attribute (default ``"round_robin"``).
    """

    BALANCE_STRATEGY = "round_robin"  # round_robin | batch_split | random

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load_context(self, context):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.models = {}
        self.tokenizers = {}

        for name, model_id in MODEL_INSTANCES.items():
            self.tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
            self.models[name] = (
                AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    add_pooling_layer=False,
                )
                .to("cuda")
                .eval()
            )

        # Round-robin state
        self._rr_index = 0
        self._instance_names = list(MODEL_INSTANCES.keys())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        texts = model_input["text"].tolist()

        # Resolve text_type
        if "text_type" in model_input.columns:
            text_types = model_input["text_type"].fillna("document").tolist()
        else:
            text_types = ["document"] * len(texts)

        # Resolve model_name per row (NaN → None for load balancing)
        if "model_name" in model_input.columns:
            model_names = [
                None if pd.isna(v) else v
                for v in model_input["model_name"].tolist()
            ]
        else:
            model_names = [None] * len(texts)

        # Apply load-balancing strategy for rows without explicit model_name
        model_names = self._apply_balancing(model_names)

        # Validate
        invalid = {m for m in model_names if m not in self.models}
        if invalid:
            valid = ", ".join(sorted(self.models.keys()))
            raise ValueError(
                f"Unknown model_name(s): {sorted(invalid)}. Valid: {valid}"
            )

        # Group rows by target instance and run batched inference
        results = [None] * len(texts)

        for target in self.models:
            indices = [i for i, mn in enumerate(model_names) if mn == target]
            if not indices:
                continue

            batch_texts = [texts[i] for i in indices]
            batch_types = [text_types[i] for i in indices]

            prefixed = [
                f"query: {t}" if tt == "query" else t
                for t, tt in zip(batch_texts, batch_types)
            ]

            embeddings = self._embed_batch(target, prefixed)

            for idx, emb in zip(indices, embeddings):
                results[idx] = {
                    "embedding": json.dumps(emb.tolist()),
                    "model_name": target,
                }

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Load-balancing strategies
    # ------------------------------------------------------------------

    def _apply_balancing(self, model_names: list) -> list:
        """Fill ``None`` entries using the configured strategy."""
        strategy = self.BALANCE_STRATEGY
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown BALANCE_STRATEGY: {strategy}. "
                f"Valid: {', '.join(VALID_STRATEGIES)}"
            )

        nones = [i for i, m in enumerate(model_names) if m is None]
        if not nones:
            return model_names

        if strategy == "round_robin":
            return self._balance_round_robin(model_names, nones)
        elif strategy == "batch_split":
            return self._balance_batch_split(model_names, nones)
        else:  # random
            return self._balance_random(model_names, nones)

    def _balance_round_robin(self, model_names, none_indices):
        """Assign unrouted rows to instances in alternating order."""
        out = list(model_names)
        for i in none_indices:
            out[i] = self._instance_names[self._rr_index % len(self._instance_names)]
            self._rr_index += 1
        return out

    def _balance_batch_split(self, model_names, none_indices):
        """Split unrouted rows evenly across instances."""
        out = list(model_names)
        mid = len(none_indices) // 2
        for i in none_indices[:mid]:
            out[i] = self._instance_names[0]
        for i in none_indices[mid:]:
            out[i] = self._instance_names[1]
        return out

    def _balance_random(self, model_names, none_indices):
        """Assign unrouted rows randomly."""
        out = list(model_names)
        for i in none_indices:
            out[i] = random.choice(self._instance_names)
        return out

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_batch(self, model_name: str, texts: list[str]) -> np.ndarray:
        """Tokenize, forward-pass, CLS-pool, and L2-normalise."""
        import torch
        import torch.nn.functional as F

        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encoded)

        cls_embeddings = outputs.last_hidden_state[:, 0]
        normalised = F.normalize(cls_embeddings, p=2, dim=1)

        return normalised.cpu().float().numpy()
