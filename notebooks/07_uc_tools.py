# Databricks notebook source
# MAGIC %md
# MAGIC # Step 7 - Agent tools as Unity Catalog functions
# MAGIC The agents never write raw SQL against tables or call the vector index directly. They call these
# MAGIC **governed, reusable functions** in `wb_demo.ai`:
# MAGIC
# MAGIC | Function | Used by | Returns |
# MAGIC |---|---|---|
# MAGIC | `list_indicators()` | Data Analyst agent | available indicators and their meaning |
# MAGIC | `get_indicator(country, indicator, start_year, end_year)` | Data Analyst agent | one country's time series |
# MAGIC | `compare_countries(countries, indicator, start_year, end_year)` | Data Analyst agent | several countries side by side + period average + rank |
# MAGIC | `get_country_profile(country)` | Data Analyst agent | latest value of every indicator for a country |
# MAGIC | `search_reports(query, country)` | Research agent | top report passages with title, page and link for citations |
# MAGIC
# MAGIC Why UC functions: one definition shared by agents, Genie, SQL users and apps; `EXECUTE` grants
# MAGIC control who may call them; every call is in the audit log; lineage links tool → table → agent.
# MAGIC **The function and parameter comments matter** - the LLM reads them to decide which tool to call and how.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG}")
C = CATALOG
log_event("uc_tools", "started")

# Reusable SQL predicates: match a country by ISO-3 or by (start of) name, and an indicator by column name, WB code or name.
# Names are compared letters-only, so "Vietnam" matches the World Bank's "Viet Nam" and "Egypt" matches "Egypt, Arab Rep.";
# a few common names that differ entirely from the World Bank's spelling are mapped explicitly.
NAME_ALIASES = {"turkey": "TUR", "türkiye": "TUR", "turkiye": "TUR", "vietnam": "VNM", "egypt": "EGY", "usa": "USA",
                "united states of america": "USA", "us": "USA", "america": "USA"}

def letters(expr: str) -> str:
    return f"lower(regexp_replace({expr}, '[^A-Za-z]', ''))"

def country_match(param: str, alias: str = "l", with_region: bool = False) -> str:
    alias_case = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in NAME_ALIASES.items())
    # Name-prefix matching only for inputs longer than an ISO code - otherwise 'IND' would also match Indonesia
    cond = (f"(upper({alias}.country_iso3) = upper(trim({param})) "
            f"OR (length(trim({param})) > 3 AND {letters(f'{alias}.country_name')} LIKE concat({letters(param)}, '%')) "
            f"OR {alias}.country_iso3 = CASE lower(trim({param})) {alias_case} END")
    if with_region:
        cond += f" OR (length(trim({param})) > 3 AND {letters(f'{alias}.region')} LIKE concat({letters(param)}, '%'))"
    return cond + ")"

INDICATOR_MATCH = ("(lower(l.column_name) = lower(trim(indicator)) OR upper(d.indicator_id) = upper(trim(indicator)) "
                   "OR lower(d.indicator_name) LIKE lower(concat('%', trim(indicator), '%')))")

# COMMAND ----------

# MAGIC %md ## Long-format view the numeric tools read from

# COMMAND ----------

unpivot_cols = ", ".join(INDICATORS.values())
spark.sql(f"""
CREATE OR REPLACE VIEW {C}.gold.indicator_values_long
COMMENT 'gold.country_year_facts in long format: one row per country, year and indicator (nulls dropped). Backs the ai.* agent tools.'
AS SELECT country_iso3, country_name, region, income_group, year, column_name, value
FROM gold.country_year_facts
UNPIVOT (value FOR column_name IN ({unpivot_cols}))
""")

# COMMAND ----------

# MAGIC %md ## Data Analyst tools

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {C}.ai.list_indicators()
RETURNS TABLE (indicator STRING, indicator_name STRING, definition STRING)
COMMENT 'Lists every development indicator available (GDP, growth, poverty, inflation, population, life expectancy, literacy, CO2, electricity access, trade, FDI, government debt). Call this first when unsure which indicator name to pass to the other tools.'
RETURN
  SELECT column_name, indicator_name, left(definition, 300)
  FROM {C}.gold.indicator_definitions
  ORDER BY column_name
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {C}.ai.get_indicator(
  country    STRING COMMENT 'Country name or ISO-3 code, e.g. "India", "IND", "Kenya"',
  indicator  STRING COMMENT 'Indicator column name from list_indicators, e.g. "gdp_growth_pct", "poverty_rate_pct", "inflation_pct". A World Bank code or part of the indicator name also works.',
  start_year INT DEFAULT 2000 COMMENT 'First year to return (inclusive), e.g. 2010',
  end_year   INT DEFAULT 2100 COMMENT 'Last year to return (inclusive)'
)
RETURNS TABLE (country_name STRING, country_iso3 STRING, year INT, indicator STRING, indicator_name STRING, value DOUBLE)
COMMENT 'Returns the yearly time series of one World Bank indicator for one country. Use for questions like "What was Kenya''s GDP growth in 2019?" or "How has India''s inflation changed since 2015?". Percent indicators are already in percent.'
RETURN
  SELECT l.country_name, l.country_iso3, l.year, l.column_name, d.indicator_name, round(l.value, 4)
  FROM {C}.gold.indicator_values_long l
  JOIN {C}.gold.indicator_definitions d ON d.column_name = l.column_name
  WHERE {country_match("country")}
    AND {INDICATOR_MATCH}
    AND l.year BETWEEN coalesce(start_year, 1960) AND coalesce(end_year, 2100)
  ORDER BY l.column_name, l.year
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {C}.ai.compare_countries(
  countries  STRING COMMENT 'Comma-separated country names or ISO-3 codes, e.g. "India, Bangladesh" or "IND,BGD,PAK"',
  indicator  STRING COMMENT 'Indicator column name from list_indicators, e.g. "gdp_growth_pct"',
  start_year INT DEFAULT 2000 COMMENT 'First year to compare (inclusive)',
  end_year   INT DEFAULT 2100 COMMENT 'Last year to compare (inclusive)'
)
RETURNS TABLE (year INT, country_name STRING, indicator STRING, indicator_name STRING, value DOUBLE,
               period_avg DOUBLE, rank_in_year INT)
COMMENT 'Compares one World Bank indicator across several countries year by year. Also returns each country''s average over the period (period_avg) and its rank among the compared countries each year (1 = highest). Use for "How has India''s GDP growth compared to Bangladesh since 2010?".'
RETURN
  WITH wanted AS (SELECT trim(c) AS c FROM (SELECT explode(split(countries, ',')) AS c) WHERE trim(c) <> ''),
  vals AS (
    SELECT l.year, l.country_name, l.column_name, d.indicator_name, l.value
    FROM {C}.gold.indicator_values_long l
    JOIN {C}.gold.indicator_definitions d ON d.column_name = l.column_name
    LEFT SEMI JOIN wanted w ON {country_match("w.c")}
    WHERE {INDICATOR_MATCH}
      AND l.year BETWEEN coalesce(start_year, 1960) AND coalesce(end_year, 2100)
  )
  SELECT year, country_name, column_name, indicator_name, round(value, 4),
         round(avg(value) OVER (PARTITION BY country_name, column_name), 4),
         CAST(rank() OVER (PARTITION BY year, column_name ORDER BY value DESC) AS INT)
  FROM vals
  ORDER BY column_name, year, country_name
""")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {C}.ai.get_country_profile(
  country STRING COMMENT 'Country name or ISO-3 code, e.g. "Nigeria" or "NGA"'
)
RETURNS TABLE (country_name STRING, region STRING, income_group STRING, indicator STRING, indicator_name STRING,
               latest_value DOUBLE, latest_year INT)
COMMENT 'Returns the most recent available value of every indicator for one country, plus its region and income group. Use for overview questions like "Give me an economic snapshot of Nigeria".'
RETURN
  SELECT l.country_name, l.region, l.income_group, l.column_name, d.indicator_name,
         round(max_by(l.value, l.year), 4), max(l.year)
  FROM {C}.gold.indicator_values_long l
  JOIN {C}.gold.indicator_definitions d ON d.column_name = l.column_name
  WHERE {country_match("country")}
  GROUP BY ALL
  ORDER BY l.column_name
""")

# COMMAND ----------

# MAGIC %md ## Research tool - wraps the Vector Search index
# MAGIC `vector_search()` is a SQL table function, so retrieval is also a governed UC function. The country
# MAGIC (or region) is added to the query text - chunks carry a "Title | Scope | Date" header, so this steers
# MAGIC retrieval toward that country's reports - and then used as a filter on the candidates.

# COMMAND ----------

def search_reports_sql(query_arg: str, hybrid: bool) -> str:
    # HYBRID = semantic + keyword: report names and specific terms ("automation", "Africa's Pulse") match exactly
    query_type = "query_type => 'HYBRID'," if hybrid else ""
    return f"""
CREATE OR REPLACE FUNCTION {C}.ai.search_reports(
  query   STRING COMMENT 'What to look for in World Bank reports, phrased as a topic, e.g. "main risks to growth outlook", "drivers of inflation", "poverty reduction policies"',
  country STRING DEFAULT NULL COMMENT 'Optional country name, ISO-3 code or World Bank region (e.g. "Kenya", "IND", "South Asia"). Leave NULL to search all reports.'
)
RETURNS TABLE (title STRING, country_name STRING, region STRING, doc_year INT, page_start INT, page_end INT,
               landing_url STRING, chunk_text STRING, score DOUBLE)
COMMENT 'Searches World Bank country and regional reports (Economic/Development Updates, Economic Focus, Africa''s Pulse, Global Economic Prospects, Poverty Assessments, Climate & Development Reports) and returns the most relevant passages with title, pages and URL for citation. Use for "why", "what does the World Bank say", risks, outlook and policy questions.'
RETURN
  SELECT title, country_name, region, CAST(doc_year AS INT), CAST(page_start AS INT), CAST(page_end AS INT),
         landing_url, chunk_text, CAST(search_score AS DOUBLE)
  FROM vector_search(
    index => '{VS_INDEX}',
    {query_arg} => CASE WHEN country IS NULL OR trim(country) = '' THEN query ELSE concat(trim(country), ': ', query) END,
    {query_type}
    num_results => 60
  ) l
  WHERE country IS NULL OR trim(country) = '' OR {country_match("country", "l", with_region=True)}
  ORDER BY search_score DESC
  LIMIT 8
"""

# Try the richest variant first; fall back if this runtime's vector_search() lacks an argument
for query_arg, hybrid in [("query_text", True), ("query_text", False), ("query", False)]:
    try:
        spark.sql(search_reports_sql(query_arg, hybrid))
        spark.sql(f"SELECT * FROM {C}.ai.search_reports('growth outlook', 'Kenya')").limit(1).collect()   # must also run
        print(f"✓ ai.search_reports created ({'hybrid' if hybrid else 'semantic'} search, `{query_arg}` argument)")
        break
    except Exception as e:
        print(f"· {query_arg}/{'hybrid' if hybrid else 'ann'} not supported → {str(e).splitlines()[0][:120]}")

# COMMAND ----------

# MAGIC %md ## Test every tool - the way the agents will call them

# COMMAND ----------

tests = {
    "list_indicators":     f"SELECT * FROM {C}.ai.list_indicators()",
    "get_indicator":       f"SELECT * FROM {C}.ai.get_indicator('Kenya', 'gdp_growth_pct', 2019, 2019)",
    "compare_countries":   f"SELECT * FROM {C}.ai.compare_countries('India, Bangladesh', 'gdp_growth_pct', 2010, 2025)",
    "get_country_profile": f"SELECT * FROM {C}.ai.get_country_profile('NGA')",
    "search_reports":      f"SELECT title, region, doc_year, page_start, score, left(chunk_text, 200) AS preview FROM {C}.ai.search_reports('main risks to growth outlook', 'South Asia')",
    "search_reports (no country)": f"SELECT title, country_name, doc_year, score FROM {C}.ai.search_reports('impact of climate change on agriculture')",
    # Regression tests: WB spellings ("Viet Nam", "Turkiye") and ISO codes must resolve to exactly one country
    "alias Vietnam/Turkey": f"""SELECT count(DISTINCT country_name) AS n FROM (
                                  SELECT country_name FROM {C}.ai.get_country_profile('Vietnam')
                                  UNION ALL SELECT country_name FROM {C}.ai.get_country_profile('Turkey'))
                                HAVING n = 2""",
    "IND is India only":  f"""SELECT count(DISTINCT country_name) AS n FROM {C}.ai.get_country_profile('IND') HAVING n = 1""",
}
results = {}
for name, q in tests.items():
    try:
        df = spark.sql(q)
        results[name] = df.count()
        print(f"✓ {name:<28} {results[name]} rows")
        display(df.limit(12))
    except Exception as e:
        results[name] = f"ERROR: {str(e).splitlines()[0][:200]}"
        print(f"✗ {name:<28} {results[name]}")

# COMMAND ----------

# MAGIC %md ## Governance

# COMMAND ----------

stmts = [
    # Schema-level EXECUTE was granted in Step 1; function-level grants show fine-grained control
    f"GRANT EXECUTE ON FUNCTION {C}.ai.search_reports TO `{GROUP_AI_APP}`",
    f"GRANT EXECUTE ON FUNCTION {C}.ai.compare_countries TO `{GROUP_ANALYSTS}`",
]
for stmt in stmts:
    try:
        spark.sql(stmt)
        print("✓", stmt)
    except Exception as e:
        print("⚠", stmt, "→", str(e).splitlines()[0][:150])

# The functions, their comments and permissions: Catalog → wb_demo → ai → Functions
# (SHOW/DESCRIBE FUNCTION output can't be rendered on serverless - UNSUPPORTED_DATATYPE - so it isn't printed here)

# COMMAND ----------

# MAGIC %md
# MAGIC **Governance demo idea:** `wb_analysts` has `EXECUTE` on `ai` but no `SELECT` on the vector index, so an
# MAGIC analyst can run `ai.compare_countries(...)` but `ai.search_reports(...)` is denied - and that denial shows up
# MAGIC in `system.access.audit` (Step 15).

# COMMAND ----------

failed = [k for k, v in results.items() if isinstance(v, str) or v == 0]
log_event("uc_tools", "complete", level="WARNING" if failed else "INFO", results=results)
assert not failed, f"Tools with errors or no rows: {failed}"
