# main.py
# Streamlit + Snowflake end-to-end demo.
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
# ceates the page title and uses a full width layout
st.set_page_config(page_title="Missile AI", layout="centered")

# ---------- UI styling and theme set up ----------
st.markdown("""
<style>
/* Centered, narrow app container */
[data-testid="block-container"]{
  max-width: 960px !important;   /* change this if you want the whole page wider/narrower */
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Section titles like: Upload CSV, Analyze & Impute, etc. */
.section-title{
  text-align: center;
  font-weight: 900;
  /* Bigger: grows a bit on larger screens */
  font-size: clamp(28px, 3.2vw, 40px);
  line-height: 1.12;
  letter-spacing: .2px;
  margin: .6rem 0 .5rem;
}

/* Make ALL Streamlit buttons larger (text + hit area) */
.stButton > button{
  font-size: 20px !important;          /* bigger button text */
  font-weight: 700 !important;
  padding: 0.9rem 1.4rem !important;   /* taller/wider buttons */
  border-radius: 12px !important;
}

/* File uploader tweaks: slightly bigger label & helper text */
[data-testid="stFileUploader"] label{
  font-size: 16px; 
  font-weight: 600;
}
[data-testid="stFileUploader"] small{
  font-size: 14px;
}

/* Optional: make Streamlit subheaders (st.subheader) a bit bigger */
h3, .stMarkdown h3{
  font-size: 28px;
  line-height: 1.2;
}
</style>
""", unsafe_allow_html=True)

## set my variables for the DB/schema/tables
DB = "DEMO_DB"
RAW = "RAW"
PROC = "PROC"
STAGING = f"{DB}.{RAW}.STAGING"
WORKING = f"{DB}.{PROC}.WORKING"
SUMMARY = f"{DB}.{PROC}.SUMMARY"
CLEANED = f"{DB}.{PROC}.CLEANED"

# ---------- LLM settings (single model) ----------
ALLOWED_METHODS = {"mean", "median", "mode", "constant", "drop"}
# You can set `st.secrets["cortex_models"]["model"]` to override
SINGLE_MODEL = st.secrets.get("cortex_models", {}).get("model", "mistral-large")

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

# create the SnowPark session ; needed to auto-create the Staging table to match the pandas DF
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
# Here: ID (int), COL_A (str | None), COL_B (float | None)

# declaring the expected col and respective types
class UploadRow(BaseModel):
    ID: int
    COL_A: t.Optional[str] = None
    COL_B: t.Optional[float] = None

# allows the cleaning of raw values before checking type
    @field_validator("COL_A", mode="before")
    def cast_str(cls, v):
        if pd.isna(v):
            return None
        return str(v)

def validate_df(df: pd.DataFrame) -> t.Tuple[bool, str]:
    """Validate DataFrame rows against the UploadRow model."""
    # enforces the exact columns from the expected tables
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

# ---------- Snowflake Key Words utilities ----------
# create the staging table if it not does not exist
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
# grab the column names and column types
def get_columns_and_types(conn) -> pd.DataFrame:
    sql = f"""
      SELECT COLUMN_NAME, DATA_TYPE
      FROM {DB}.INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = '{RAW}' AND TABLE_NAME = 'STAGING'
      ORDER BY ORDINAL_POSITION
    """
    return pd.read_sql(sql, conn)

# count the IS NULL and calc a missing rate & create the Summary Table
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

# --- Single-model Cortex helper ---
def _llm_complete(conn, model: str, prompt: str) -> str:
    """
    Try AI_COMPLETE first; if not available in the account, try SNOWFLAKE.CORTEX.COMPLETE.
    Returns raw text response.
    """
    # Attempt AI_COMPLETE(model => ..., prompt => ...)
    try:
        df = pd.read_sql(
            "SELECT AI_COMPLETE(model => %s, prompt => %s) AS RESP",
            conn,
            params=[model, prompt],
        )
        return str(df.iloc[0, 0]).strip()
    except Exception:
        pass

    # Attempt SNOWFLAKE.CORTEX.COMPLETE(model, prompt)
    try:
        df = pd.read_sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS RESP",
            conn,
            params=[model, prompt],
        )
        return str(df.iloc[0, 0]).strip()
    except Exception as e:
        raise e

def _normalize_method(text: t.Optional[str]) -> t.Optional[str]:
    """
    Normalize any LLM response to one of ALLOWED_METHODS.
    Accepts extra words like 'Use median.' and pulls the keyword out.
    """
    if not text:
        return None
    tkn = str(text).strip().lower()
    for m in ALLOWED_METHODS:
        if m in tkn:
            return m
    return None

def pick_method_llm(conn, col_name: str, miss: int, dtype: str) -> str:
    """
    Ask ONE Cortex model for an imputation method.
    Falls back to a simple heuristic if the call fails or returns junk.
    """
    prompt = (
        "Return ONE word for the best imputation method for a column with missing values. "
        f"Allowed words: {', '.join(sorted(ALLOWED_METHODS))}. "
        f"Column={col_name}, Missing={miss}, Type={dtype}. "
        "Answer with ONE word only."
    )
    try:
        raw = _llm_complete(conn, SINGLE_MODEL, prompt)
        m = _normalize_method(raw)
        if m:
            return m
    except Exception:
        pass

    # fallback heuristic
    if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
        return "median"
    return "mode"

# creates a table with the imputations applied of the uploaded table
def apply_imputations(conn, use_llm: bool) -> pd.DataFrame:
    """
    Create working table and impute nulls column-by-column.
    If use_llm=True, pick imputation via the single Cortex model.
    Returns a summary DataFrame with chosen methods.
    """
    # fresh working copy
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE {WORKING} AS SELECT * FROM {STAGING}")

    # columns/types from STAGING
    cols = pd.read_sql(
        f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM {DB}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'STAGING'
        ORDER BY ORDINAL_POSITION
        """,
        conn,
        params=[RAW],
    )

    summary_rows = []

    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        dtype = r["DATA_TYPE"]

        # how many are missing?
        miss = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        if miss == 0:
            summary_rows.append([col, miss, 0.0, "none"])
            continue

        if use_llm:
            method = pick_method_llm(conn, col, miss, dtype)
        else:
            method = "median" if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]) else "mode"

        # ---- apply the chosen method ----
        if method == "drop":
            cur.execute(f"DELETE FROM {WORKING} WHERE {col} IS NULL")
        elif method in {"mean", "median"} and any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
            agg = "AVG" if method == "mean" else "MEDIAN"
            val = cur.execute(f"SELECT {agg}({col}) FROM {WORKING} WHERE {col} IS NOT NULL").fetchone()[0]
            cur.execute(f"UPDATE {WORKING} SET {col} = %s WHERE {col} IS NULL", (val,))
        elif method == "mode":
            res = cur.execute(
                f"""
                SELECT {col}
                FROM {WORKING}
                WHERE {col} IS NOT NULL
                GROUP BY {col}
                ORDER BY COUNT(*) DESC
                LIMIT 1
                """
            ).fetchone()
            fill = res[0] if res else "UNKNOWN"
            cur.execute(f"UPDATE {WORKING} SET {col} = %s WHERE {col} IS NULL", (fill,))
        elif method == "constant":
            cur.execute(f"UPDATE {WORKING} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")
        else:
            # safety net
            cur.execute(f"UPDATE {WORKING} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")

        # recompute rate after imputation
        total_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
        missing_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        rate_after = (missing_after / total_after) if total_after else 0.0

        summary_rows.append([col, miss, rate_after, method])

    # save cleaned table
    cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")

    # write summary
    sdf = pd.DataFrame(
        summary_rows,
        columns=["COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER", "IMPUTATION_METHOD"],
    )
    cur.execute(
        f"""
        CREATE OR REPLACE TABLE {SUMMARY} (
          COLUMN_NAME STRING,
          MISSING_BEFORE NUMBER,
          MISSING_RATE_AFTER FLOAT,
          IMPUTATION_METHOD STRING
        )
        """
    )
    write_pandas(conn, sdf, table_name="SUMMARY", database=DB, schema=PROC, quote_identifiers=False)

    return sdf

###### ---------- UI ----------

## title text and centering it
st.markdown(
    "<h1 style='text-align:center; font-size:72px; line-height:1.06; margin:0.2em 0 48px;'>Missile AI</h1>",
    unsafe_allow_html=True,
)

conn = get_connector()
ensure_objects(conn)

## upload CSV process
st.markdown("<div class='center-narrow'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Upload CSV</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv"])  # empty label; we use our own title
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

    if ok and st.button("Upload to Snowflake (STAGING)", use_container_width=True):
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

st.markdown("</div>", unsafe_allow_html=True)  # close center-narrow
st.divider()

### Analysis and Imputation buttons
st.markdown("<div class='center-narrow'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Analyze & Impute</div>", unsafe_allow_html=True)

# two buttons side-by-side, centered by the narrow container
bcol1, bcol2 = st.columns(2, gap="large")

with bcol1:
    if st.button("Analyze Missing (no LLM)", use_container_width=True):
        summary = build_missing_summary(conn)
        if summary.empty:
            st.warning("STAGING is empty.")
        else:
            st.subheader("Missing Summary (STAGING)")
            st.dataframe(summary)

with bcol2:
    if st.button("Analyze + Impute (use Cortex if available)", use_container_width=True):
        try:
            sdf = apply_imputations(conn, use_llm=True)
            st.success("Imputation completed. See tables below.")
            st.subheader("Imputation Summary")
            st.dataframe(sdf)
        except Exception as e:
            st.error("Imputation failed.")
            st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)  # close center-narrow
st.divider()

## Show results / show tables section
st.markdown("<div class='section-title'>Preview Table and Action Summary</div>", unsafe_allow_html=True)
st.markdown("<div class='center-narrow'>", unsafe_allow_html=True)

if st.button("View Tables", use_container_width=True):
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

st.markdown("</div>", unsafe_allow_html=True)  # close center-narrow
st.divider()

### Finalize or cleanup
st.markdown("<div class='center-narrow'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Finish</div>", unsafe_allow_html=True)

# One-click publish: finalize & clean PROC artifacts
if st.button("Publish (Finalize & Clean)", use_container_width=True):
    cur = conn.cursor()
    try:
        # Ensure CLEANED exists (user must have run an Impute step)
        cur.execute(f"SELECT 1 FROM {CLEANED} LIMIT 1")

        # Finalize: promote CLEANED to RAW.FINAL
        cur.execute(f"CREATE OR REPLACE TABLE {DB}.{RAW}.FINAL AS SELECT * FROM {CLEANED}")

        # Clean: drop working PROC tables
        cur.execute(f"DROP TABLE IF EXISTS {SUMMARY}")
        cur.execute(f"DROP TABLE IF EXISTS {WORKING}")
        cur.execute(f"DROP TABLE IF EXISTS {CLEANED}")

        st.success(f"Published to {DB}.{RAW}.FINAL and cleaned PROC tables.")
    except snowflake.connector.errors.ProgrammingError as e:
        st.error(
            "Publish failed. Make sure you've run an Impute step (so PROC.CLEANED exists) "
            "and that your role has CREATE TABLE on DEMO_DB.RAW."
        )
        st.exception(e)
    except Exception as e:
        st.error("Publish failed.")
        st.exception(e)

# Small advanced expander for rare “start fresh” need
with st.expander("Advanced: Reset Pipeline (danger)"):
    st.caption("Drops PROC tables and empties RAW.STAGING so you can upload a fresh CSV.")
    if st.button("Reset Pipeline"):
        try:
            cur = conn.cursor()
            cur.execute(f"DROP TABLE IF EXISTS {SUMMARY}")
            cur.execute(f"DROP TABLE IF EXISTS {WORKING}")
            cur.execute(f"DROP TABLE IF EXISTS {CLEANED}")
            cur.execute(f"TRUNCATE TABLE IF EXISTS {STAGING}")
            st.warning("Pipeline reset: dropped PROC tables and truncated RAW.STAGING.")
        except Exception as e:
            st.error("Reset failed.")
            st.exception(e)

st.markdown("</div>", unsafe_allow_html=True)
