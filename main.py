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
import re

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
SINGLE_MODEL = st.secrets.get("cortex_models", {}).get("model", "reka-flash")

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

def _sf_identifier(name: str) -> str:
    """Make a Snowflake-safe identifier: UPPERCASE, only letters/digits/_ and unique."""
    s = re.sub(r'[^0-9A-Za-z_]+', '_', str(name).strip())
    if re.match(r'^[0-9]', s):  # identifier can't start with a digit
        s = "_" + s
    return s.upper()

def _sf_type_from_dtype(dtype) -> str:
    import pandas as pd
    from pandas.api import types as ptypes
    if ptypes.is_integer_dtype(dtype):
        return "NUMBER"
    if ptypes.is_float_dtype(dtype):
        return "FLOAT"
    if ptypes.is_bool_dtype(dtype):
        return "BOOLEAN"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP_NTZ"
    return "TEXT"  # fallback for strings/mixed

def validate_df_any(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame, list[tuple[str, str]]]:
    """
    Permissive check:
    - not empty
    - unique, Snowflake-safe column names (we'll rename if needed)
    - cells are scalars (no lists/dicts)
    Returns (ok, msg, cleaned_df, renames).
    """
    if df is None or df.empty:
        return False, "CSV is empty.", df, []

    clean = df.copy()
    original = list(clean.columns)

    # sanitize and make unique
    new_cols, seen, renames = [], set(), []
    for col in original:
        nc = _sf_identifier(col)
        base = nc
        i = 1
        while nc in seen:
            i += 1
            nc = f"{base}_{i}"
        seen.add(nc)
        new_cols.append(nc)
        if str(col) != nc:
            renames.append((str(col), nc))
    clean.columns = new_cols

    # basic "pydantic-like" scalar check on a sample
    for v in clean.head(200).to_numpy().flatten():
        if isinstance(v, (list, dict, set, tuple)):
            return False, "Cells must be text/numbers/booleans/timestamps (no arrays/objects).", clean, renames

    return True, "OK", clean, renames

def create_or_replace_staging_for_df(conn, df: pd.DataFrame, full_table: str):
    """Generate DDL to match the DataFrame and CREATE OR REPLACE the STAGING table."""
    cols = [f'{c} {_sf_type_from_dtype(df[c].dtype)}' for c in df.columns]
    ddl = ", ".join(cols)
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE {full_table} ({ddl})")

# ---------- Snowflake Key Words utilities ----------
# create the staging table if it not does not exist
def ensure_objects(conn):
  return

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
    Try AI_COMPLETE first; if not available, try SNOWFLAKE.CORTEX.COMPLETE.
    Returns raw text response.
    """
    try:
        df = pd.read_sql(
            "SELECT AI_COMPLETE(model => %s, prompt => %s) AS RESP",
            conn,
            params=[model, prompt],
        )
        return str(df.iloc[0, 0]).strip()
    except Exception:
        pass

    df = pd.read_sql(
        "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS RESP",
        conn,
        params=[model, prompt],
    )
    return str(df.iloc[0, 0]).strip()

def _normalize_method(text: str | None) -> str | None:
    if not text:
        return None
    tkn = str(text).strip().lower()
    # accept sentences like "Use median"
    for m in ALLOWED_METHODS:
        if m in tkn:
            return m
    return None
  
def choose_methods_llm_bulk(conn, profile: list[dict]) -> dict[str, str]:
    """
    ONE Cortex call for all columns with missing data.
    profile = [{name, type, missing_count, total_rows, missing_rate}, ...]
    Returns a dict: {column_name -> method}
    """
    # Only send columns that have missing > 0 and are not clearly numeric (we'll default numeric to median fast)
    ai_candidates = []
    for p in profile:
        dt = p["type"]
        if p["missing_count"] > 0 and not any(x in dt for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
            ai_candidates.append({
                "name": p["name"],
                "type": dt,
                "missing_count": p["missing_count"],
                "missing_rate": round(p["missing_rate"], 6),
            })

    # If nothing to ask the LLM, return empty (we'll fill heuristically)
    if not ai_candidates:
        return {}

    # Keep prompt compact; ask for strict JSON with allowed methods only
    prompt = (
        "Return a STRICT JSON object mapping each column name to ONE imputation method word. "
        f"Allowed words only: {', '.join(sorted(ALLOWED_METHODS))}. "
        "Use 'median' for numeric-like data (not included here), 'mode' or 'constant' for text; "
        "use 'drop' only if missing_rate > 0.5 and dropping won't destroy the table. "
        "No prose, no markdown, just JSON. "
        f"Columns: {json.dumps(ai_candidates, ensure_ascii=False)}"
    )

    raw = _llm_complete(conn, SINGLE_MODEL, prompt)
    # Try to extract JSON
    try:
        # Strip code fences if any
        if raw.strip().startswith("```"):
            raw = raw.split("```")[1]
        plan = json.loads(raw)
        # Validate / normalize methods
        out = {}
        for k, v in plan.items():
            m = _normalize_method(v)
            if m:
                out[k] = m
        return out
    except Exception:
        # If parsing failed, return empty -> we’ll do heuristics
        return {}

# creates a table with the imputations applied of the uploaded table
def apply_imputations(conn, use_llm: bool) -> pd.DataFrame:
    """
    Create working table and impute nulls column-by-column.
    If use_llm=True, call Cortex ONCE to get a JSON plan for all columns.
    Returns a summary DataFrame with chosen methods.
    """
    cur = conn.cursor()
    # Fresh working copy
    cur.execute(f"CREATE OR REPLACE TABLE {WORKING} AS SELECT * FROM {STAGING}")

    # Build a missing profile once
    cols_df = pd.read_sql(
        f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM {DB}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'STAGING'
        ORDER BY ORDINAL_POSITION
        """,
        conn,
        params=[RAW],
    )
    if cols_df.empty:
        st.warning("No columns found in STAGING.")
        return pd.DataFrame(columns=["COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER", "IMPUTATION_METHOD"])

    total = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0] or 0

    profile = []
    for _, r in cols_df.iterrows():
        col = r["COLUMN_NAME"]
        dtype = r["DATA_TYPE"]
        miss = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        rate = (miss / total) if total else 0.0
        profile.append({"name": col, "type": dtype, "missing_count": miss, "total_rows": total, "missing_rate": rate})

    # Plan methods: one LLM call for all text-like columns
    llm_plan: dict[str, str] = {}
    if use_llm:
        llm_plan = choose_methods_llm_bulk(conn, profile)

    summary_rows = []

    # Apply imputations
    for p in profile:
        col = p["name"]
        dtype = p["type"]
        miss = p["missing_count"]

        if miss == 0:
            summary_rows.append([col, 0, 0.0, "none"])
            continue

        # Heuristic defaults (fast)
        if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
            method = "median"
        else:
            method = llm_plan.get(col, "mode" if p["missing_rate"] < 0.5 else "constant")

        # ---- apply chosen method ----
        if method == "drop":
            # Note: this deletes rows with NULL in this column
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
        new_total = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
        missing_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        rate_after = (missing_after / new_total) if new_total else 0.0
        summary_rows.append([col, miss, rate_after, method])

    # Save cleaned table & write summary
    cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")

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

uploaded = st.file_uploader("", type=["csv"])  # label is above in your title
if uploaded:
    raw_df = pd.read_csv(uploaded)

    ok, msg, df_clean, renames = validate_df_any(raw_df)

    st.subheader("Preview:")
    st.dataframe(df_clean.head())

    if renames:
        st.caption("Renamed for Snowflake: " + ", ".join([f"{a} → {b}" for a, b in renames]))

    if ok:
        st.success("Basic validation: OK")
        if st.button("Upload to Snowflake (STAGING)", use_container_width=True):
            # Build STAGING to match the CSV
            create_or_replace_staging_for_df(conn, df_clean, STAGING)

            # Load data (quote_identifiers=True since our cols may not be bare keywords)
            write_pandas(
                conn,
                df_clean,
                table_name="STAGING",
                database=DB,
                schema=RAW,
                quote_identifiers=True,
                overwrite=True,  # replace data on re-upload
            )
            st.success(f"Uploaded {len(df_clean):,} rows into {STAGING}")
    else:
        st.error(msg)

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



