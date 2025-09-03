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
import json
import math
import numpy as np
_NUMERIC_MARKERS = ("NUMBER", "FLOAT", "DOUBLE", "DECIMAL", "INT")

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

/* keep all buttons consistent and centered when used alone */
.stButton > button { min-width: 220px; }

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

/* --- add these --- */
.subtitle{
  text-align:center;
  font-weight:700;
  font-size: clamp(16px, 2.2vw, 22px);
  line-height:1.3;
  opacity:.9;
  margin:-4px 0 10px;   /* tight to the title */
}
.spacer-xxl{ height: 72px; }  /* “good chunk” of space */

/* Big responsive video */
.video-wrap{
  display:flex; justify-content:center; margin: 14px auto 0;
}
.video-16x9{
  width:100%; max-width: 980px; aspect-ratio: 16 / 9;
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
}
.video-16x9 iframe{ width:100%; height:100%; border:0; }

/* Larger gap before the CSV section */
.spacer-hero{ height: 110px; }

/* Darker background gradient (no white) */
:root{
  /* tweak these three to taste */
  --bg-top:    #0b0d10;  /* near-black */
  --bg-mid:    #12171d;  /* dark slate */
  --bg-bottom: #1b222a;  /* slightly lighter, still dark */
}

.stApp{
  background: linear-gradient(
    180deg,
    var(--bg-top) 0%,
    var(--bg-mid) 45%,
    var(--bg-bottom) 100%
  ) !important;
}

/* keep header/sidebar transparent so gradient shows through */
[data-testid="stHeader"], [data-testid="stSidebar"]{
  background: transparent !important;
  backdrop-filter: none !important;
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


def _is_numeric_dtype_name(dtype_name: str) -> bool:
    dn = (dtype_name or "").upper()
    return any(m in dn for m in _NUMERIC_MARKERS)

def _jsonify(val):
    """Make any scalar json-serializable; map NaN to None."""
    if val is None:
        return None
    if isinstance(val, (np.generic,)):  # numpy scalar -> python scalar
        return val.item()
    if isinstance(val, float) and math.isnan(val):
        return None
    return val  # ints/str/bool already fine

def _parse_llm_methods(resp_text: str) -> dict[str, str]:
    """
    Expect the model to return a JSON array like:
      [{"name":"COL_A","method":"median"}, ...]
    We still try to be forgiving and extract the first [...] block.
    """
    s = (resp_text or "").strip()
    # Extract first JSON array if the model wrapped text around it
    start = s.find("[")
    end   = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    data = json.loads(s)  # will raise if it's not valid JSON
    out: dict[str, str] = {}
    for obj in data:
        # accept common key variants
        name = (obj.get("name") or obj.get("column") or obj.get("col") or "").strip()
        method = (obj.get("method") or "").strip().lower()
        if name and method in ALLOWED_METHODS:
            out[name.upper()] = method
    return out
  
def choose_methods_llm_bulk(conn, profile_df: pd.DataFrame, model: str | None = None) -> dict[str, str]:
    """
    profile_df columns expected: COLUMN_NAME, DATA_TYPE, TOTAL_ROWS, MISSING_COUNT
    Returns mapping: {COLUMN_NAME -> method}
    """
    if model is None:
        # single-model path (fast) – reuse whatever you configured as your primary model
        model = SINGLE_MODEL

    # Keep only columns that have missing values; cap to keep prompt small/fast
    prof = profile_df.copy()
    prof["MISSING_COUNT"] = prof["MISSING_COUNT"].fillna(0).astype(int)
    prof = prof[prof["MISSING_COUNT"] > 0].sort_values("MISSING_COUNT", ascending=False)

    if prof.empty:
        return {}

    # reasonable cap (tune if needed)
    prof = prof.head(40)

    # Build a clean JSON payload (plain types only)
    payload = []
    dtype_map: dict[str, str] = {}
    for _, r in prof.iterrows():
        col = str(r["COLUMN_NAME"]).upper()
        dtype = str(r["DATA_TYPE"])
        total = int(_jsonify(r.get("TOTAL_ROWS", 0)))
        miss  = int(_jsonify(r.get("MISSING_COUNT", 0)))
        dtype_map[col] = dtype
        payload.append({
            "name": col,
            "dtype": dtype,
            "total": total,
            "missing": miss,
            "numeric": _is_numeric_dtype_name(dtype)
        })

    prompt = (
        "You are selecting an imputation method for EACH column.\n"
        "Valid methods: mean, median, mode, constant, drop.\n"
        "Rules of thumb:\n"
        "- numeric columns usually use median (robust) or mean\n"
        "- categorical text uses mode; use constant only if very sparse\n"
        "- use drop only if the column is mostly missing and non-critical\n\n"
        "Return ONLY valid JSON array: "
        '[{"name":"<COLUMN_NAME>","method":"<one of mean|median|mode|constant|drop>"}].\n\n'
        f"Columns: {json.dumps(payload, ensure_ascii=False)}"
    )

    methods: dict[str, str] = {}
    try:
        resp = _llm_complete(conn, model, prompt)
        methods = _parse_llm_methods(resp)
    except Exception:
        methods = {}

    # Fallback for anything missing from the LLM output
    for col in (p["name"] for p in payload):
        if col not in methods:
            methods[col] = "median" if _is_numeric_dtype_name(dtype_map[col]) else "mode"

    return methods

# creates a table with the imputations applied of the uploaded table
def apply_imputations(conn, use_llm: bool) -> pd.DataFrame:
    """
    Create WORKING from STAGING, impute nulls, save CLEANED + SUMMARY.
    If use_llm=True, use choose_methods_llm_bulk() to pick per-column methods;
    otherwise use a simple heuristic (median for numeric, mode for text).
    """
    def qident(name: str) -> str:
        # Quote an identifier safely for Snowflake (handles spaces/mixed case)
        return '"' + str(name).replace('"', '""') + '"'

    cur = conn.cursor()

    # Fresh working copy
    cur.execute(f"CREATE OR REPLACE TABLE {WORKING} AS SELECT * FROM {STAGING}")

    # Discover columns and types from STAGING (same layout as WORKING right now)
    cols = pd.read_sql(
        f"""
        SELECT COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
        FROM {DB}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'STAGING'
        ORDER BY ORDINAL_POSITION
        """,
        conn,
        params=[RAW],
    )

    # Nothing to do?
    if cols.empty:
        cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")
        sdf = pd.DataFrame(
            columns=[
                "COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER",
                "IMPUTATION_METHOD", "SOURCE", "VOTES"
            ]
        )
        cur.execute(
            f"""
            CREATE OR REPLACE TABLE {SUMMARY} (
              COLUMN_NAME STRING,
              MISSING_BEFORE NUMBER,
              MISSING_RATE_AFTER FLOAT,
              IMPUTATION_METHOD STRING,
              SOURCE STRING,
              VOTES STRING
            )
            """
        )
        if not sdf.empty:
            write_pandas(conn, sdf, table_name="SUMMARY", database=DB, schema=PROC, quote_identifiers=False)
        return sdf

    # Totals
    total_rows = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
    if not total_rows:
        # Empty table; just materialize CLEANED and empty SUMMARY
        cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")
        sdf = pd.DataFrame(
            columns=[
                "COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER",
                "IMPUTATION_METHOD", "SOURCE", "VOTES"
            ]
        )
        cur.execute(
            f"""
            CREATE OR REPLACE TABLE {SUMMARY} (
              COLUMN_NAME STRING,
              MISSING_BEFORE NUMBER,
              MISSING_RATE_AFTER FLOAT,
              IMPUTATION_METHOD STRING,
              SOURCE STRING,
              VOTES STRING
            )
            """
        )
        return sdf

    # Build a quick profile (missing counts per column)
    prof_rows = []
    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        miss = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {qident(col)} IS NULL").fetchone()[0]
        prof_rows.append([col, r["DATA_TYPE"], int(total_rows), int(miss)])
    profile = pd.DataFrame(prof_rows, columns=["COLUMN_NAME", "DATA_TYPE", "TOTAL_ROWS", "MISSING_COUNT"])

    # Ask the single LLM (bulk) only once, if requested
    llm_plan: dict[str, str] = {}
    if use_llm:
        try:
            llm_plan = choose_methods_llm_bulk(conn, profile, model=SINGLE_MODEL)  # {COL -> method}
        except Exception:
            llm_plan = {}

    # Apply per column
    summary_rows = []

    for _, r in cols.iterrows():
        col = str(r["COLUMN_NAME"])
        dt  = str(r["DATA_TYPE"])
        col_q = qident(col)

        # missing before
        miss_before = int(profile.loc[profile["COLUMN_NAME"] == col, "MISSING_COUNT"].values[0])

        if miss_before == 0:
            summary_rows.append([col, 0, 0.0, "none", "none", ""])
            continue

        # Decide method
        if use_llm:
            method = llm_plan.get(col.upper())
            if not method:
                method = "median" if _is_numeric_dtype_name(dt) else "mode"
                source = "heuristic"
            else:
                source = "llm"
        else:
            method = "median" if _is_numeric_dtype_name(dt) else "mode"
            source = "heuristic"

        # --- Apply method ---
        if method == "drop":
            cur.execute(f"DELETE FROM {WORKING} WHERE {col_q} IS NULL")

        elif method in ("mean", "median") and _is_numeric_dtype_name(dt):
            agg = "AVG" if method == "mean" else "MEDIAN"
            val = cur.execute(
                f"SELECT {agg}({col_q}) FROM {WORKING} WHERE {col_q} IS NOT NULL"
            ).fetchone()[0]
            # Only update if we have a value
            if val is not None:
                cur.execute(f"UPDATE {WORKING} SET {col_q} = %s WHERE {col_q} IS NULL", (val,))

        elif method == "mode":
            res = cur.execute(
                f"""
                SELECT {col_q}
                FROM {WORKING}
                WHERE {col_q} IS NOT NULL
                GROUP BY {col_q}
                ORDER BY COUNT(*) DESC, {col_q}
                LIMIT 1
                """
            ).fetchone()
            if res is None:
                # No non-null values; choose a sensible constant
                fill = 0 if _is_numeric_dtype_name(dt) else "UNKNOWN"
            else:
                fill = res[0]
            cur.execute(f"UPDATE {WORKING} SET {col_q} = %s WHERE {col_q} IS NULL", (fill,))

        elif method == "constant":
            fill = 0 if _is_numeric_dtype_name(dt) else "UNKNOWN"
            cur.execute(f"UPDATE {WORKING} SET {col_q} = %s WHERE {col_q} IS NULL", (fill,))

        else:
            # Safety net: behave like 'mode' for text, 'median' for numeric
            if _is_numeric_dtype_name(dt):
                val = cur.execute(
                    f"SELECT MEDIAN({col_q}) FROM {WORKING} WHERE {col_q} IS NOT NULL"
                ).fetchone()[0]
                if val is not None:
                    cur.execute(f"UPDATE {WORKING} SET {col_q} = %s WHERE {col_q} IS NULL", (val,))
            else:
                res = cur.execute(
                    f"""
                    SELECT {col_q}
                    FROM {WORKING}
                    WHERE {col_q} IS NOT NULL
                    GROUP BY {col_q}
                    ORDER BY COUNT(*) DESC, {col_q}
                    LIMIT 1
                    """
                ).fetchone()
                fill = (res[0] if res else "UNKNOWN")
                cur.execute(f"UPDATE {WORKING} SET {col_q} = %s WHERE {col_q} IS NULL", (fill,))

        # Stats after
        total_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
        miss_after  = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col_q} IS NULL").fetchone()[0]
        rate_after  = (miss_after / total_after) if total_after else 0.0

        summary_rows.append([col, miss_before, rate_after, method, source, ""])

    # Materialize CLEANED and SUMMARY
    cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")

    sdf = pd.DataFrame(
        summary_rows,
        columns=[
            "COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER",
            "IMPUTATION_METHOD", "SOURCE", "VOTES"
        ],
    )

    cur.execute(
        f"""
        CREATE OR REPLACE TABLE {SUMMARY} (
          COLUMN_NAME STRING,
          MISSING_BEFORE NUMBER,
          MISSING_RATE_AFTER FLOAT,
          IMPUTATION_METHOD STRING,
          SOURCE STRING,
          VOTES STRING
        )
        """
    )
    write_pandas(conn, sdf, table_name="SUMMARY", database=DB, schema=PROC, quote_identifiers=False)

    return sdf

###### ---------- UI ----------

st.markdown(
    "<h1 style='text-align:center; font-size:72px; line-height:1.06; margin:0.2em 0 8px;'>Missile AI</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>AI powered app that fills in missing data in tables</p>",
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ------- Show/orial Video -------
# ---- Tutorial (toggle + looping embed that never reaches the end) ----
VIDEO_ID = "krCbTpvFqEs"          
START_AT = 0                      
END_AT   = 143

# build a privacy-friendly embed that loops between START_AT and END_AT
yt_src = (
    f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}"
    f"?start={START_AT}&end={END_AT}"
    f"&loop=1&playlist={VIDEO_ID}"        # loop requires playlist=VIDEO_ID
    f"&rel=0&modestbranding=1&iv_load_policy=3&playsinline=1"
)

# default: hidden
if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = False

left, center, right = st.columns([1,1,1])
with center:
  if st.button("Watch Tutorial", key="watch_tutorial"):
      st.session_state.show_tutorial = not st.session_state.show_tutorial

# a little gap under the button
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# only render the video when toggled on
if st.session_state.show_tutorial:
        st.markdown(
            f"""
            <div class="video-16x9" style="margin:0 auto; max-width:980px; border-radius:12px; overflow:hidden;
                 box-shadow:0 10px 30px rgba(0,0,0,.35)">
              <iframe src={yt_src} title="Tutorial"
                allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # smaller "Hide" button to collapse, keeps the main label fixed
    # hide_l, hide_c, hide_r = st.columns([1,1,1])
    # with hide_c:
    #     if st.button("Hide Tutorial", key="hide_tutorial_button"):
    #         st.session_state.show_tutorial = False

# extra space before the CSV section
st.markdown("<div class='spacer-hero'></div>", unsafe_allow_html=True)



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
st.markdown("<div class='section-title'>Analyze & Fill Data</div>", unsafe_allow_html=True)

# two buttons side-by-side, centered by the narrow container
bcol1, bcol2 = st.columns(2, gap="large")

with bcol1:
    if st.button("Count Missing Data", use_container_width=True):
        summary = build_missing_summary(conn)
        if summary.empty:
            st.warning("STAGING is empty.")
        else:
            st.subheader("Missing Summary (STAGING)")
            st.dataframe(summary)

with bcol2:
    if st.button("Count & Fill Missing Data", use_container_width=True):
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
st.markdown("<div class='section-title'>Publish Table</div>", unsafe_allow_html=True)

# One-click publish: finalize & clean PROC artifacts
if st.button("Push To Database", use_container_width=True):
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























