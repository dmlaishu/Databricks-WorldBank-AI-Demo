from databricks import sql
import os

CATALOG = os.getenv("CATALOG", "world_bank_ai")

def build_indicator_query(country_code: str, indicator_code: str, limit: int = 5) -> str:
    allowed_countries = {"IND", "USA", "CHN"}
    allowed_indicators = {
        "NY.GDP.MKTP.KD.ZG",
        "SP.POP.TOTL",
        "SL.UEM.TOTL.ZS",
        "FP.CPI.TOTL.ZG",
    }
    if country_code not in allowed_countries or indicator_code not in allowed_indicators:
        raise ValueError("Unsupported country or indicator")
    return f"""
      SELECT country_name, indicator_name, year, value
      FROM {CATALOG}.gold.country_indicators
      WHERE country_code = '{country_code}'
        AND indicator_code = '{indicator_code}'
      ORDER BY year DESC
      LIMIT {int(limit)}
    """

def run_sql_query(query: str):
    server_hostname = os.environ["DATABRICKS_SERVER_HOSTNAME"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    token = os.environ.get("DATABRICKS_TOKEN")
    with sql.connect(server_hostname=server_hostname, http_path=http_path, access_token=token) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
