# main.py
# Streamlit + Snowflake end-to-end demo (no AWS).
# - Upload CSV
# - Validate with Pydantic
# - Write to Snowflake
# - Analyze nulls & (optionally) use Snowflake Cortex to pick imputation methods
# - Apply imputations in a working table and show both Summary + Cleaned

import os
import typing as t

import pandas as pd
import streamlit as st
from pydantic import BaseModel, field_validator

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark import Session

# ---------- App config ----------
st.set_page_config(page_title="Missile AI Tool", layout="wide")

DB = "DEMO_DB"
RAW = "RAW"
PROC = "PROC"
STAGING = f"{DB}.{RAW}.STAGING"
WORKING = f"{DB}.{PROC}.WORKING"
SUMMARY = f"{DB}.{PROC}.SUMMARY"
CLEANED = f"{DB}.{PROC}.CLEANED"

# ---------- Helpers: connections ----------

@st.cache_resource
def get_connector():
    """Snowflake Python connector connection (cached)."""
    cfg = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        account=str(cfg["account"]).strip(),
        user=str(cfg["user"]).strip(),
        password=cfg["password"],
        authenticator="snowflake",
        client_session_keep_alive=True,
    )
    cur = conn.cursor()
    # set context after connect so errors are obvious
    cur.execute(f"USE ROLE {st.secrets['snowflake'].get('role', 'APP_ROLE')}")
    cur.execute(f"USE WAREHOUSE {st.secrets['snowflake'].get('warehouse', 'DEMO_WH')}")
    cur.execute(f"USE DATABASE {st.secrets['snowflake'].get('database', DB)}")
    cur.execute(f"USE SCHEMA {st.secrets['snowflake'].get('schema', RAW)}")
    return conn


@st.cache_resource
def get_session():
    """Snowpark Session (handy for auto_create_table on write_pandas)."""
    cfg = st.secrets["snowflake"]
    return Session.builder.configs(
        {
            "account": str(cfg["account"]).strip(),
            "user": str(cfg["user"]).strip(),
            "password": cfg["password"],
            "role": cfg.get("role", "APP_ROLE"),
            "warehouse": cfg.get("warehouse", "DEMO_WH"),
            "database": cfg.get("database", DB),
            "schema": cfg.get("schema", RAW),
        }
    ).create()

# ---------- Schema validation with Pydantic ----------
# Adjust the model to your expected upload schema.
# Here: ID (int), COL_A (str), COL_B (float | None)

class UploadRow(BaseModel):
    ID: int
    COL_A: t.Optional[str] = None
    COL_B: t.Optional[float] = None

    @field_validator("COL_A", mode="before")
    def cast_str(cls, v):
        if pd.isna(v):
            return None
        return str(v)

def validate_df(df: pd.DataFrame) -> t.Tuple[bool, str]:
    """Validate DataFrame rows against the UploadRow model."""
    required_cols = {"ID", "COL_A", "COL_B"}
    if set(map(str.upper, df.columns)) != required_cols:
        return False, f"CSV columns must be exactly {sorted(required_cols)} (case-insensitive)."

    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    try:
        # Validate a small sample first for speed
        sample = df.head(200)
        for rec in sample.to_dict(orient="records"):
            UploadRow(**rec)
    except Exception as e:
        return False, f"Pydantic validation failed: {e}"
    return True, "OK"

# ---------- Snowflake DDL utilities ----------

def ensure_objects(conn):
    cur = conn.cursor()
    # Assume DB/SCHEMAs already exist (created by admin).
    # Only create the working tables the app needs.

    # RAW.STAGING (typed or generic; adjust to your CSV)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB}.{RAW}.STAGING (
            ID NUMBER,
            COL_A STRING,
            COL_B FLOAT,
            _LOAD_TS TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

# ---------- Analysis & Imputation ----------

def get_columns_and_types(conn) -> pd.DataFrame:
    sql = f"""
      SELECT COLUMN_NAME, DATA_TYPE
      FROM {DB}.INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = '{RAW}' AND TABLE_NAME = 'STAGING'
      ORDER BY ORDINAL_POSITION
    """
    return pd.read_sql(sql, conn)

def build_missing_summary(conn) -> pd.DataFrame:
    """Compute missing counts per column from STAGING."""
    cols = get_columns_and_types(conn)
    if cols.empty:
        return pd.DataFrame(columns=["COLUMN_NAME", "TOTAL_ROWS", "MISSING_COUNT", "MISSING_RATE"])

    cur = conn.cursor()
    total = cur.execute(f"SELECT COUNT(*) FROM {STAGING}").fetchone()[0]
    rows = []
    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        miss = cur.execute(
            f"SELECT COUNT(*) FROM {STAGING} WHERE {col} IS NULL"
        ).fetchone()[0]
        rate = (miss / total) if total else 0
        rows.append([col, total, miss, rate])
    return pd.DataFrame(rows, columns=["COLUMN_NAME", "TOTAL_ROWS", "MISSING_COUNT", "MISSING_RATE"])

def pick_method_with_cortex(conn, col_name: str, miss: int, dtype: str) -> str:
    """
    Ask Snowflake Cortex (AI_COMPLETE) for an imputation method.
    Falls back to a sensible default if AI is unavailable.
    """
    prompt = (
        "You are selecting a single best imputation method for a column with missing values. "
        "Valid answers: mean, median, mode, constant, drop. "
        f"Column name: {col_name}. Missing count: {miss}. Data type: {dtype}. "
        "Return just one word from the valid answers."
    )
    try:
        sql = f"""
          SELECT AI_COMPLETE(
            model => 'mistral-large',
            prompt => %s
          ) AS METHOD
        """
        method = pd.read_sql(sql, conn, params=[prompt]).iloc[0, 0]
        method = str(method).strip().lower()
        if method not in {"mean", "median", "mode", "constant", "drop"}:
            raise ValueError("invalid method")
        return method
    except Exception:
        # fallback heuristic
        if "NUMBER" in dtype or "FLOAT" in dtype or "INT" in dtype or "DECIMAL" in dtype or "DOUBLE" in dtype:
            return "median"
        return "mode"

def create_working_from_staging(conn):
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE {WORKING} AS SELECT * FROM {STAGING}")

def apply_imputations(conn, use_llm: bool) -> pd.DataFrame:
    """
    Create working table and impute nulls column-by-column.
    Returns a summary DataFrame with chosen methods.
    """
    create_working_from_staging(conn)
    cols = get_columns_and_types(conn)
    cur = conn.cursor()
    summary_rows = []

    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        dtype = r["DATA_TYPE"]
        miss = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        if miss == 0:
            summary_rows.append([col, miss, 0.0, "none"])
            continue

        method = pick_method_with_cortex(conn, col, miss, dtype) if use_llm else (
            "median" if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]) else "mode"
        )

        # Apply chosen method
        if method == "drop":
            cur.execute(f"DELETE FROM {WORKING} WHERE {col} IS NULL")
        elif method in ("mean", "median") and any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
            agg = "AVG" if method == "mean" else "MEDIAN"
            # compute fill value
            val = cur.execute(f"SELECT {agg}({col}) FROM {WORKING} WHERE {col} IS NOT NULL").fetchone()[0]
            cur.execute(f"UPDATE {WORKING} SET {col} = %s WHERE {col} IS NULL", (val,))
        elif method == "mode":
            mode_sql = f"""
              SELECT {col}
              FROM {WORKING}
              WHERE {col} IS NOT NULL
              GROUP BY {col}
              ORDER BY COUNT(*) DESC
              LIMIT 1
            """
            res = cur.execute(mode_sql).fetchone()
            fill = res[0] if res else "UNKNOWN"
            cur.execute(f"UPDATE {WORKING} SET {col} = %s WHERE {col} IS NULL", (fill,))
        elif method == "constant":
            cur.execute(f"UPDATE {WORKING} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")
        else:
            # fallback safety
            cur.execute(f"UPDATE {WORKING} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")

        # recompute rate on working
        total = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
        missing_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        rate = (missing_after / total) if total else 0.0
        summary_rows.append([col, miss, rate, method])

    # save cleaned table snapshot
    cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")

    # write summary
    sdf = pd.DataFrame(summary_rows, columns=["COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER", "IMPUTATION_METHOD"])
    cur.execute(f"CREATE OR REPLACE TABLE {SUMMARY} (COLUMN_NAME STRING, MISSING_BEFORE NUMBER, MISSING_RATE_AFTER FLOAT, IMPUTATION_METHOD STRING)")
    write_pandas(conn, sdf, table_name="SUMMARY", database=DB, schema=PROC, quote_identifiers=False)

    return sdf

# ---------- UI ----------

st.title("Missile AI Tool")

conn = get_connector()
ensure_objects(conn)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
df_uploaded: t.Optional[pd.DataFrame] = None

if uploaded:
    df_uploaded = pd.read_csv(uploaded)
    st.subheader("Preview:")
    st.dataframe(df_uploaded.head())

    ok, msg = validate_df(df_uploaded)
    if ok:
        st.success("Pydantic validation: OK")
    else:
        st.error(msg)

    if ok and st.button("Upload to Snowflake (STAGING)"):
        # normalize column names to match STAGING
        df_to_write = df_uploaded.copy()
        df_to_write.columns = [c.upper() for c in df_to_write.columns]

        # easiest: let Snowpark auto-create (in case STAGING was adjusted)
        session = get_session()
        session.write_pandas(
            df_to_write,
            table_name="STAGING",
            database=DB,
            schema=RAW,
            auto_create_table=True,
            overwrite=False,
        )
        st.success(f"Uploaded {len(df_to_write):,} rows into {STAGING}")

st.divider()

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    if st.button("Analyze Missing (no LLM)"):
        summary = build_missing_summary(conn)
        if summary.empty:
            st.warning("STAGING is empty.")
        else:
            st.subheader("Missing Summary (STAGING)")
            st.dataframe(summary)

with col_b:
    if st.button("Analyze + Impute (use Cortex if available)"):
        try:
            sdf = apply_imputations(conn, use_llm=True)
            st.success("Imputation completed. See tables below.")
            st.subheader("Imputation Summary")
            st.dataframe(sdf)
        except Exception as e:
            st.error("Imputation failed.")
            st.exception(e)

with col_c:
    if st.button("Analyze + Impute (no LLM)"):
        try:
            sdf = apply_imputations(conn, use_llm=False)
            st.success("Imputation (no LLM) completed. See tables below.")
            st.subheader("Imputation Summary")
            st.dataframe(sdf)
        except Exception as e:
            st.error("Imputation failed.")
            st.exception(e)

st.divider()

# Show results
if st.button("Show Tables"):
    try:
        st.subheader("SUMMARY (PROC.SUMMARY)")
        try:
            df_sum = pd.read_sql(f"SELECT * FROM {SUMMARY} ORDER BY COLUMN_NAME", conn)
            st.dataframe(df_sum)
        except Exception:
            st.info("SUMMARY not found yet.")

        st.subheader("CLEANED (PROC.CLEANED) sample")
        try:
            df_clean = pd.read_sql(f"SELECT * FROM {CLEANED} LIMIT 200", conn)
            st.dataframe(df_clean)
        except Exception:
            st.info("CLEANED not found yet.")
    except Exception as e:
        st.error("Could not fetch tables.")
        st.exception(e)

st.divider()

# Finalize or cleanup
fc1, fc2, fc3 = st.columns([1,1,1])

with fc1:
    if st.button("Finalize: Copy CLEANED → RAW.FINAL"):
        cur = conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB}.{RAW}")
        cur.execute(f"CREATE OR REPLACE TABLE {DB}.{RAW}.FINAL AS SELECT * FROM {CLEANED}")
        st.success(f"Created {DB}.{RAW}.FINAL from CLEANED")

with fc2:
    if st.button("Delete PROC tables (SUMMARY, WORKING, CLEANED)"):
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {SUMMARY}")
        cur.execute(f"DROP TABLE IF EXISTS {WORKING}")
        cur.execute(f"DROP TABLE IF EXISTS {CLEANED}")
        st.warning("Deleted PROC tables.")

with fc3:
    if st.button("Truncate STAGING"):
        conn.cursor().execute(f"TRUNCATE TABLE IF EXISTS {STAGING}")
        st.warning("STAGING truncated.")

