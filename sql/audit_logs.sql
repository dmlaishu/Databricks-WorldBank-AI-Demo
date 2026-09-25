-- Unity Catalog/system-table audit example.
-- Availability and exact schema depend on workspace/account configuration.
-- Run after system tables are enabled for the workspace.

SELECT
  event_time,
  user_identity.email AS user_email,
  service_name,
  action_name,
  request_params,
  response
FROM system.access.audit
WHERE event_date >= current_date() - INTERVAL 7 DAYS
ORDER BY event_time DESC
LIMIT 100;

-- Application telemetry created by notebook 01.
SELECT *
FROM world_bank_ai.monitoring.app_logs
ORDER BY event_time DESC
LIMIT 100;
