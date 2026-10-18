-- =====================================================================
-- Step 5 - Databricks SQL analytics on the governed gold layer
-- Open: SQL Editor → paste → pick the Serverless Starter Warehouse → run one query at a time.
-- Save each as a query (⋮ → Save) to show "saved queries" in the demo.
-- =====================================================================

USE CATALOG wb_demo;

-- 1. The demo question, first half: India vs Bangladesh GDP growth since 2010
SELECT year,
       max(CASE WHEN country_iso3 = 'IND' THEN round(gdp_growth_pct, 2) END) AS india_gdp_growth,
       max(CASE WHEN country_iso3 = 'BGD' THEN round(gdp_growth_pct, 2) END) AS bangladesh_gdp_growth
FROM gold.country_year_facts
WHERE country_iso3 IN ('IND', 'BGD') AND year >= 2010
GROUP BY year
ORDER BY year;

-- 2. Average GDP growth by region over time (countries in scope)
SELECT region, year, round(avg(gdp_growth_pct), 2) AS avg_gdp_growth_pct, count(gdp_growth_pct) AS countries
FROM gold.country_year_facts
WHERE year >= 2005
GROUP BY ALL
ORDER BY region, year;

-- 3. Poverty vs life expectancy - latest available year per country
SELECT country_name, region, income_group,
       poverty_rate_pct, poverty_rate_pct_year,
       life_expectancy_years, life_expectancy_years_year
FROM gold.latest_country_snapshot
WHERE poverty_rate_pct IS NOT NULL
ORDER BY poverty_rate_pct DESC;

-- 4. GDP per capita ranking with 10-year change
SELECT country_name, region,
       round(max(CASE WHEN year = 2014 THEN gdp_per_capita_usd END)) AS gdp_pc_2014,
       round(max(CASE WHEN year = 2024 THEN gdp_per_capita_usd END)) AS gdp_pc_2024,
       round(100 * (max(CASE WHEN year = 2024 THEN gdp_per_capita_usd END)
                  / max(CASE WHEN year = 2014 THEN gdp_per_capita_usd END) - 1), 1) AS pct_change
FROM gold.country_year_facts
GROUP BY ALL
ORDER BY gdp_pc_2024 DESC NULLS LAST;

-- 5. Window function: 3-year rolling average inflation, South Asia
SELECT country_name, year, round(inflation_pct, 1) AS inflation_pct,
       round(avg(inflation_pct) OVER (PARTITION BY country_iso3 ORDER BY year ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 1) AS inflation_3yr_avg
FROM gold.country_year_facts
WHERE region = 'South Asia' AND year >= 2015
ORDER BY country_name, year;

-- 6. Structured + unstructured side by side: numbers and how many report chunks back them
SELECT s.country_name, s.region,
       round(s.gdp_growth_pct, 2) AS latest_gdp_growth, s.gdp_growth_pct_year AS year,
       count(DISTINCT c.doc_id) AS reports, count(c.chunk_id) AS report_chunks, max(c.doc_date) AS newest_report
FROM gold.latest_country_snapshot s
LEFT JOIN gold.doc_chunks c ON c.country_iso3 = s.country_iso3
GROUP BY ALL
ORDER BY reports DESC, country_name;

-- 7. Governance in SQL: what does each column mean, and where did it come from?
SELECT column_name, indicator_id, indicator_name, left(definition, 200) AS definition
FROM gold.indicator_definitions
ORDER BY column_name;

-- 8. Delta time travel on gold
DESCRIBE HISTORY gold.country_year_facts;
SELECT count(*) AS rows_in_first_version FROM gold.country_year_facts VERSION AS OF 0;

-- 9. AI functions inside SQL (Foundation Model API from a query)
--    Replace the endpoint name with a chat model listed under Serving if this one is not available.
SELECT country_name,
       ai_query('databricks-meta-llama-3-3-70b-instruct',
                concat('In one sentence, describe this economy: ', to_json(struct(*)))) AS one_liner
FROM gold.latest_country_snapshot
WHERE country_iso3 IN ('IND', 'KEN')
LIMIT 2;
