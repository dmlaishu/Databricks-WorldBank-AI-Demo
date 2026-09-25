-- Structured demo: India's latest GDP growth observations
SELECT
  country_name,
  year,
  value AS gdp_growth_pct
FROM world_bank_ai.gold.country_indicators
WHERE country_code = 'IND'
  AND indicator_code = 'NY.GDP.MKTP.KD.ZG'
ORDER BY year DESC
LIMIT 5;

-- Compare latest GDP growth observations across demo countries
WITH ranked AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY country_code ORDER BY year DESC) AS rn
  FROM world_bank_ai.gold.country_indicators
  WHERE indicator_code = 'NY.GDP.MKTP.KD.ZG'
)
SELECT country_name, year, value AS gdp_growth_pct
FROM ranked
WHERE rn <= 5
ORDER BY country_name, year DESC;
