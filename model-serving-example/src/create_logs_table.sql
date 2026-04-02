-- Create the telemetry target table for recommendation predictions.
-- Zerobus does NOT create tables; this must run before enabling ingestion.
-- event_timestamp is LONG (not TIMESTAMP) because Zerobus requires integer microseconds.

CREATE TABLE IF NOT EXISTS main.model_serving.recommendation_logs (
    request_id        STRING    NOT NULL COMMENT 'UUID for deduplication',
    category          STRING    NOT NULL COMMENT 'Input category',
    match_type        STRING    NOT NULL COMMENT 'exact, partial, or default',
    recommended_items STRING    NOT NULL COMMENT 'JSON array of recommendations',
    event_timestamp   LONG      NOT NULL COMMENT 'Unix epoch microseconds'
)
COMMENT 'Prediction telemetry from recommendation model serving endpoint';

-- Grant the Zerobus service principal access.
-- Replace <service-principal-id> with the actual application ID.
-- Schema-level inherited grants are insufficient for Zerobus authorization_details OAuth flow.
GRANT USE CATALOG ON CATALOG main TO `<service-principal-id>`;
GRANT USE SCHEMA ON SCHEMA main.model_serving TO `<service-principal-id>`;
GRANT MODIFY, SELECT ON TABLE main.model_serving.recommendation_logs TO `<service-principal-id>`;
