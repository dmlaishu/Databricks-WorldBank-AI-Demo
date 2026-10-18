# Databricks notebook source
# MAGIC %md
# MAGIC # Step 5 - Databricks SQL: view + AI/BI dashboard
# MAGIC 1. Creates `gold.latest_country_snapshot` - latest non-null value of every indicator per country
# MAGIC    (used by the dashboard, ad-hoc SQL, and later by the SQL agent).
# MAGIC 2. Creates and publishes the **World Bank Development Overview** AI/BI dashboard on a SQL warehouse.
# MAGIC    Re-running replaces it, so the dashboard is code, not clicks.
# MAGIC 3. Genie space setup instructions (UI) are at the bottom.
# MAGIC
# MAGIC The same governed gold tables serve BI users here and the AI agents later.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
from databricks.sdk import WorkspaceClient

spark.sql(f"USE CATALOG {CATALOG}")
w = WorkspaceClient()
DASHBOARD_NAME = "World Bank Development Overview"
log_event("sql_dashboard", "started")

# COMMAND ----------

# MAGIC %md ## 1. Latest-value snapshot view

# COMMAND ----------

latest_cols = ",\n  ".join(
    f"max_by({c}, CASE WHEN {c} IS NOT NULL THEN year END) AS {c},\n  max(CASE WHEN {c} IS NOT NULL THEN year END) AS {c}_year"
    for c in INDICATORS.values()
)
spark.sql(f"""
CREATE OR REPLACE VIEW gold.latest_country_snapshot
COMMENT 'One row per country with the most recent non-null value of every indicator. <indicator>_year says which year each value is from.'
AS SELECT country_iso3, country_name, region, income_group,
  {latest_cols}
FROM gold.country_year_facts
GROUP BY ALL
""")
display(spark.sql("SELECT country_name, gdp_growth_pct, gdp_growth_pct_year, poverty_rate_pct, poverty_rate_pct_year FROM gold.latest_country_snapshot ORDER BY country_name"))

# COMMAND ----------

# MAGIC %md ## 2. Pick a SQL warehouse

# COMMAND ----------

warehouses = list(w.warehouses.list())
for wh in warehouses:
    print(f"{wh.name:<40} {str(wh.state).split('.')[-1]:<10} serverless={wh.enable_serverless_compute} id={wh.id}")

# Prefer a serverless warehouse; set WAREHOUSE_ID by hand if you want a specific one
WAREHOUSE_ID = next((wh.id for wh in warehouses if wh.enable_serverless_compute), warehouses[0].id if warehouses else None)
assert WAREHOUSE_ID, "No SQL warehouse found - create one under SQL Warehouses (Serverless, 2X-Small)"
print("Using warehouse", WAREHOUSE_ID)

# COMMAND ----------

# MAGIC %md ## 3. Dashboard definition
# MAGIC Datasets are plain SQL; widgets reference dataset columns. Grid is 6 columns wide.

# COMMAND ----------

C = CATALOG
datasets = {
    "kpis": f"""
        SELECT (SELECT count(DISTINCT country_iso3) FROM {C}.gold.country_year_facts) AS countries,
               (SELECT count(*) FROM {C}.gold.indicator_definitions)                  AS indicators,
               (SELECT max(year) FROM {C}.gold.country_year_facts WHERE gdp_growth_pct IS NOT NULL) AS latest_year,
               (SELECT count(DISTINCT doc_id) FROM {C}.gold.doc_chunks)               AS reports,
               (SELECT count(*) FROM {C}.gold.doc_chunks)                              AS report_chunks""",
    "region_growth": f"""
        SELECT region, year, round(avg(gdp_growth_pct), 2) AS avg_gdp_growth_pct
        FROM {C}.gold.country_year_facts WHERE year >= 2005 AND gdp_growth_pct IS NOT NULL GROUP BY ALL""",
    "south_asia_growth": f"""
        SELECT country_name, year, round(gdp_growth_pct, 2) AS gdp_growth_pct
        FROM {C}.gold.country_year_facts
        WHERE country_iso3 IN ('IND', 'BGD', 'PAK', 'LKA', 'NPL') AND year >= 2010 AND gdp_growth_pct IS NOT NULL""",
    "poverty_life": f"""
        SELECT country_name, region, round(poverty_rate_pct, 1) AS poverty_rate_pct, round(life_expectancy_years, 1) AS life_expectancy_years
        FROM {C}.gold.latest_country_snapshot WHERE poverty_rate_pct IS NOT NULL AND life_expectancy_years IS NOT NULL""",
    "gdp_per_capita": f"""
        SELECT country_name, region, round(gdp_per_capita_usd) AS gdp_per_capita_usd
        FROM {C}.gold.latest_country_snapshot WHERE gdp_per_capita_usd IS NOT NULL""",
    "reports": f"""
        SELECT coalesce(country_name, region) AS subject, scope, count(DISTINCT doc_id) AS reports, count(*) AS chunks
        FROM {C}.gold.doc_chunks GROUP BY ALL""",
}

def field(col, agg=None):
    return {"name": col, "expression": f"{agg}(`{col}`)" if agg else f"`{col}`"}

def widget(name, dataset, fields, spec, x, y, width, height):
    return {
        "widget": {
            "name": name,
            "queries": [{"name": "main_query", "query": {"datasetName": dataset, "fields": fields, "disaggregated": True}}],
            "spec": spec,
        },
        "position": {"x": x, "y": y, "width": width, "height": height},
    }

def counter(name, col, title, x):
    return widget(name, "kpis", [field(col)],
                  {"version": 2, "widgetType": "counter", "encodings": {"value": {"fieldName": col, "displayName": title}},
                   "frame": {"showTitle": True, "title": title}}, x, 2, 1, 2)

def chart(kind, name, dataset, x_col, y_col, color_col, title, x_type, pos):
    enc = {"x": {"fieldName": x_col, "scale": {"type": x_type}, "displayName": x_col},
           "y": {"fieldName": y_col, "scale": {"type": "quantitative"}, "displayName": y_col}}
    cols = [x_col, y_col]
    if color_col:
        enc["color"] = {"fieldName": color_col, "scale": {"type": "categorical"}, "displayName": color_col}
        cols.append(color_col)
    return widget(name, dataset, [field(c) for c in dict.fromkeys(cols)],
                  {"version": 3, "widgetType": kind, "encodings": enc, "frame": {"showTitle": True, "title": title}}, *pos)

layout = [
    {"widget": {"name": "header", "multilineTextboxSpec": {"lines": [
        "# World Bank Development Overview",
        f"Governed gold tables in `{C}.gold` - the same data the AI assistant uses. Source: World Bank WDI API and Documents & Reports API."]}},
     "position": {"x": 0, "y": 0, "width": 6, "height": 2}},
    counter("kpi_countries", "countries", "Countries", 0),
    counter("kpi_indicators", "indicators", "Indicators", 1),
    counter("kpi_latest_year", "latest_year", "Latest year", 2),
    counter("kpi_reports", "reports", "Reports", 3),
    counter("kpi_chunks", "report_chunks", "Report chunks (RAG)", 4),
    chart("line", "sa_growth", "south_asia_growth", "year", "gdp_growth_pct", "country_name",
          "South Asia: GDP growth % since 2010", "quantitative", (0, 4, 3, 6)),
    chart("line", "region_growth", "region_growth", "year", "avg_gdp_growth_pct", "region",
          "Average GDP growth % by region", "quantitative", (3, 4, 3, 6)),
    chart("scatter", "poverty_life", "poverty_life", "poverty_rate_pct", "life_expectancy_years", "region",
          "Poverty rate vs life expectancy (latest year)", "quantitative", (0, 10, 3, 7)),
    chart("bar", "gdp_pc", "gdp_per_capita", "country_name", "gdp_per_capita_usd", "region",
          "GDP per capita, US$ (latest year)", "categorical", (3, 10, 3, 7)),
    chart("bar", "reports_by_subject", "reports", "subject", "chunks", "scope",
          "Report chunks available to the Research agent", "categorical", (0, 17, 6, 6)),
]

serialized = json.dumps({
    "datasets": [{"name": n, "displayName": n.replace("_", " ").title(), "queryLines": [q.strip()]} for n, q in datasets.items()],
    "pages": [{"name": "overview", "displayName": "Overview", "layout": layout}],
})

# Validate every dataset query on the cluster first - a broken query fails here, not silently in the dashboard
for name, q in datasets.items():
    spark.sql(q).limit(1).collect()
print(f"{len(datasets)} dataset queries OK")

# COMMAND ----------

# MAGIC %md ## 4. Create (or replace) and publish

# COMMAND ----------

from databricks.sdk.service.dashboards import Dashboard

me = w.current_user.me().user_name
parent = f"/Users/{me}"

# Replace an earlier version of this dashboard so re-runs don't pile up copies
for obj in w.workspace.list(parent):
    if str(obj.object_type).endswith("DASHBOARD") and obj.path.split("/")[-1].startswith(DASHBOARD_NAME):
        try:
            w.lakeview.trash(obj.resource_id)
            print("Replaced previous dashboard", obj.resource_id)
        except Exception as e:
            print("Could not remove old dashboard (delete it by hand if you see two):", e)

try:   # databricks-sdk >= 0.40
    dash = w.lakeview.create(dashboard=Dashboard(display_name=DASHBOARD_NAME, parent_path=parent,
                                                 serialized_dashboard=serialized, warehouse_id=WAREHOUSE_ID))
except TypeError:   # older SDK signature
    dash = w.lakeview.create(display_name=DASHBOARD_NAME, parent_path=parent,
                             serialized_dashboard=serialized, warehouse_id=WAREHOUSE_ID)

w.lakeview.publish(dash.dashboard_id, embed_credentials=True, warehouse_id=WAREHOUSE_ID)
url = f"{w.config.host.rstrip('/')}/dashboardsv3/{dash.dashboard_id}/published"
displayHTML(f'<h3>✅ Dashboard published: <a href="{url}" target="_blank">{DASHBOARD_NAME}</a></h3>')
log_event("sql_dashboard", "complete", dashboard_id=dash.dashboard_id, warehouse_id=WAREHOUSE_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. (Optional) Genie space - natural-language SQL over gold
# MAGIC Genie is a good "BI user" demo and can later serve as the Data Analyst agent.
# MAGIC
# MAGIC 1. Sidebar → **Genie** → **New**.
# MAGIC 2. **Data**: add `wb_demo.gold.country_year_facts`, `wb_demo.gold.latest_country_snapshot`, `wb_demo.gold.indicator_definitions`.
# MAGIC 3. **Title**: `World Bank Indicators`. **Warehouse**: the serverless one used above.
# MAGIC 4. **Instructions** (paste):
# MAGIC    ```
# MAGIC    - Data is World Bank World Development Indicators for 20 countries, 2000 onwards, one row per country and year.
# MAGIC    - Growth, inflation, unemployment, poverty and *_pct columns are already percentages; do not multiply by 100.
# MAGIC    - gdp_usd is current US dollars; show it in billions (divide by 1e9) and say so.
# MAGIC    - For "latest" or "current" values use latest_country_snapshot and report the matching *_year column.
# MAGIC    - Match countries by country_name or country_iso3 (India = IND, Bangladesh = BGD).
# MAGIC    - To explain what a metric means, use indicator_definitions.
# MAGIC    ```
# MAGIC 5. **Sample questions**: *How has India's GDP growth compared to Bangladesh since 2010?* ·
# MAGIC    *Which country has the highest poverty rate?* · *Show inflation in South Asia over the last 5 years.*
# MAGIC 6. Ask one, then open **Show code** to show the generated SQL - every Genie query runs on governed tables and appears in the audit log.
