import mlflow
import pandas as pd
import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)


DEFAULT_RECOMMENDATIONS = {
    "electronics": [
        {"item": "Wireless Headphones", "score": 0.95},
        {"item": "USB-C Hub", "score": 0.88},
        {"item": "Portable Charger", "score": 0.82},
        {"item": "Bluetooth Speaker", "score": 0.75},
    ],
    "books": [
        {"item": "Python Crash Course", "score": 0.93},
        {"item": "Designing Data-Intensive Applications", "score": 0.91},
        {"item": "Clean Code", "score": 0.85},
        {"item": "The Pragmatic Programmer", "score": 0.80},
    ],
    "clothing": [
        {"item": "Running Shoes", "score": 0.90},
        {"item": "Merino Wool Socks", "score": 0.85},
        {"item": "Rain Jacket", "score": 0.78},
        {"item": "Baseball Cap", "score": 0.70},
    ],
    "home": [
        {"item": "Smart LED Bulbs", "score": 0.92},
        {"item": "French Press", "score": 0.87},
        {"item": "Throw Blanket", "score": 0.80},
        {"item": "Scented Candle", "score": 0.72},
    ],
    "sports": [
        {"item": "Yoga Mat", "score": 0.91},
        {"item": "Resistance Bands", "score": 0.86},
        {"item": "Jump Rope", "score": 0.79},
        {"item": "Foam Roller", "score": 0.73},
    ],
    "default": [
        {"item": "Gift Card", "score": 0.70},
        {"item": "Water Bottle", "score": 0.65},
        {"item": "Notebook & Pen Set", "score": 0.60},
    ],
}


class RecommendationModel(mlflow.pyfunc.PythonModel):
    """Custom pyfunc model that recommends items based on category heuristics.

    Instantiated with a recommendations dictionary mapping category names to
    lists of {item, score} dicts. At predict time, matches input categories
    using a three-tier heuristic: exact match, partial match, fallback.
    """

    def __init__(self, recommendations: dict = None):
        self.recommendations = recommendations or DEFAULT_RECOMMENDATIONS
        self._zerobus_stream = None
        self._zerobus_sdk = None
        self._zerobus_initialized = False
        self._zerobus_enabled = True

    def _init_zerobus(self):
        """Lazy one-time initialization of the Zerobus stream.

        Reads credentials from environment variables set on the serving
        endpoint. If any step fails, disables Zerobus logging silently.
        """
        if self._zerobus_initialized:
            return
        self._zerobus_initialized = True

        try:
            from zerobus.sdk.sync import ZerobusSdk
            from zerobus.sdk.shared import (
                RecordType,
                StreamConfigurationOptions,
                TableProperties,
            )

            server_endpoint = os.environ["ZEROBUS_SERVER_ENDPOINT"]
            workspace_url = os.environ["DATABRICKS_WORKSPACE_URL"]
            client_id = os.environ["DATABRICKS_CLIENT_ID"]
            client_secret = os.environ["DATABRICKS_CLIENT_SECRET"]
            table_name = os.environ.get(
                "ZEROBUS_TABLE_NAME",
                "main.model_serving.recommendation_logs",
            )

            self._zerobus_sdk = ZerobusSdk(server_endpoint, workspace_url)
            options = StreamConfigurationOptions(record_type=RecordType.JSON)
            table_props = TableProperties(table_name)
            self._zerobus_stream = self._zerobus_sdk.create_stream(
                client_id, client_secret, table_props, options
            )
            logger.info("Zerobus stream initialized for %s", table_name)
        except Exception:
            logger.warning(
                "Failed to initialize Zerobus stream. "
                "Prediction telemetry will not be logged.",
                exc_info=True,
            )
            self._zerobus_enabled = False

    def _reconnect_zerobus(self):
        """Re-create the stream after a connection failure."""
        try:
            if self._zerobus_stream is not None:
                try:
                    self._zerobus_stream.close()
                except Exception:
                    pass
                self._zerobus_stream = None

            from zerobus.sdk.shared import (
                RecordType,
                StreamConfigurationOptions,
                TableProperties,
            )

            table_name = os.environ.get(
                "ZEROBUS_TABLE_NAME",
                "main.model_serving.recommendation_logs",
            )
            client_id = os.environ["DATABRICKS_CLIENT_ID"]
            client_secret = os.environ["DATABRICKS_CLIENT_SECRET"]

            options = StreamConfigurationOptions(record_type=RecordType.JSON)
            table_props = TableProperties(table_name)
            self._zerobus_stream = self._zerobus_sdk.create_stream(
                client_id, client_secret, table_props, options
            )
            logger.info("Zerobus stream reconnected")
        except Exception:
            logger.warning(
                "Zerobus reconnection failed. Disabling telemetry.",
                exc_info=True,
            )
            self._zerobus_enabled = False

    def _log_predictions(self, log_records):
        """Ingest prediction records to Zerobus with one retry on failure."""
        if not self._zerobus_enabled or self._zerobus_stream is None:
            return

        try:
            for record in log_records:
                self._zerobus_stream.ingest_record(json.dumps(record))
            self._zerobus_stream.flush()
        except Exception as e:
            logger.warning("Zerobus ingest failed: %s", e)
            err_msg = str(e).lower()
            if "closed" in err_msg or "connection" in err_msg:
                self._reconnect_zerobus()
                try:
                    if self._zerobus_stream is not None:
                        for record in log_records:
                            self._zerobus_stream.ingest_record(json.dumps(record))
                        self._zerobus_stream.flush()
                except Exception:
                    logger.warning(
                        "Zerobus retry failed. Records dropped.",
                        exc_info=True,
                    )

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        logging.info("Predicting recommendations")
        if not self._zerobus_initialized:
            self._init_zerobus()

        results = []
        log_records = []
        now_us = int(time.time() * 1_000_000)

        # Capture OTel trace ID for correlation with MLflow traces
        trace_id = str(uuid.uuid4())
        try:
            from opentelemetry import trace as otel_trace
            span = otel_trace.get_current_span()
            span_ctx = span.get_span_context()
            if span_ctx.trace_id:
                trace_id = format(span_ctx.trace_id, '032x')
        except Exception:
            pass

        for _, row in model_input.iterrows():
            category = str(row["category"]).strip().lower()
            items, match_type = self._find_recommendations(category)
            ranked = sorted(items, key=lambda x: x["score"], reverse=True)
            recommended_json = json.dumps(ranked)

            results.append({
                "recommended_items": recommended_json,
                "match_type": match_type,
            })

            log_records.append({
                "request_id": trace_id,
                "category": category,
                "match_type": match_type,
                "recommended_items": recommended_json,
                "event_timestamp": now_us,
            })

        try:
            self._log_predictions(log_records)
        except Exception:
            logger.warning("Unexpected error in prediction logging", exc_info=True)

        return pd.DataFrame(results)

    def _find_recommendations(self, category: str):
        """Three-tier heuristic: exact -> partial -> default."""
        # Tier 1: Exact match
        if category in self.recommendations:
            return self.recommendations[category], "exact"

        # Tier 2: Partial/substring match
        for key in self.recommendations:
            if key == "default":
                continue
            if key in category or category in key:
                return self.recommendations[key], "partial"

        # Tier 3: Fallback
        return self.recommendations.get("default", []), "default"
