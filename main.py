# main.py
# Streamlit + Snowflake (no AWS) — column-agnostic version
# - Any CSV columns (sanitized to Snowflake-safe names)
# - Light Pydantic validation (values: str|int|float|None)
# - Write to Snowflake (drop/recreate RAW.STAGING per upload)
# - Analyze nulls & (optionally) use Snowflake Cortex ensemble for imputations
# - Show Summary + Cleaned; Publish to RAW.FINAL

import typing as t
import re
import numpy as np
import pandas as pd
import streamlit as st
from pydantic import RootModel

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark import Session

# ---------------- App config ----------------
st.set_page_config(page_title="Missile AI", layout="centered")

# ---------------- Styling ----------------
st.markdown("""
<style>
/* Gradient background across the whole app */
html, body, [data-testid="stAppViewContainer"] {
  height: 100%;
  background: linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 45%, #2a2a2a 100%) !important;
  background-attachment: fixed;
  color: #e5e7eb;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: rgba(0,0,0,0.35) !important; backdrop-filter: blur(2px); }
[data-testid="block-container"]{ background: transparent !important; }

/* Centered, narrow page width */
[data-testid="block-container"]{
  max-width: 960px !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Main title + subtitle */
.h1-title { text-align:center; font-size:72px; line-height:1.06; margin:0.2em 0 8px; font-weight:900; }
.subtitle  { text-align:center; font-size:20px; opacity:.9; margin-bottom: 20px; }

/* Section titles like: Upload CSV / Analyze & Impute / etc */
.section-title{
  text-align: center;
  font-weight: 900;
  font-size: clamp(28px, 3.2vw, 40px);
  line-height: 1.12;
  letter-spacing: .2px;
  margin: 28px 0 12px;
}

/* Bigger buttons (supports newer & older Streamlit selectors) */
.stButton > button,
button[kind="primary"],
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"]{
  font-size: 20px !important;
  font-weight: 800 !important;
  padding: 0.95rem 1.5rem !important;
  border-radius: 12px !important;
}

/* Inputs on dark bg */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox > div > div, .stMultiSelect > div > div {
  background-color: #111 !important;
  color: #e5e7eb !important;
  border: 1px solid #333 !important;
}

/* Dataframe headers tweak for dark theme */
[data-testid="stDataFrame"] table thead th { background: #111 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- Snowflake targets ----------------
DB   = "DEMO_DB"
RAW  = "RAW"
PROC = "PROC"

STAGING = f"{DB}.{RAW}.STAGING"
WORKING = f"{DB}.{PROC}.WORKING"
SUMMARY = f"{DB}.{PROC}.SUMMARY"
CLEANED = f"{DB}.{PROC}.CLEANED"

# ---------------- Cortex ensemble settings ----------------
ALLOWED_METHODS = {"mean", "median", "mode", "constant", "drop"}
_models = st.secrets.get("cortex_models", {})
MODEL_A = _models.get("model_a", "mistral-large")
MODEL_B = _models.get("model_b", "llama3-70b")
MODEL_C = _models.get("model_c", "reka-flash")
ARBITER_MODEL = _models.get("arbiter", "mistral-large")

# ---------------- Connections ----------------
@st.cache_resource
def get_connector():
    cfg = st.secrets["snowflake"]
    conn = snowflake.connector.connect(
        account=str(cfg["account"]).strip(),
        user=str(cfg["user"]).strip(),
        password=cfg["password"],
        authenticator="snowflake",
        client_session_keep_alive=True,
    )
    cur = conn.cursor()
    cur.execute(f"USE ROLE {cfg.get('role', 'APP_ROLE')}")
    cur.execute(f"USE WAREHOUSE {cfg.get('warehouse', 'DEMO_WH')}")
    cur.execute(f"USE DATABASE {cfg.get('database', DB)}")
    cur.execute(f"USE SCHEMA {cfg.get('schema', RAW)}")
    return conn

@st.cache_resource
def get_session():
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

# ---------------- Column & validation helpers ----------------
BasicScalar = t.Union[str, int, float, None]

class AnyRow(RootModel[dict[str, BasicScalar]]):
    """Pydantic v2 RootModel: row is {column: str|int|float|None}."""
    pass

_SAFE_START = re.compile(r'^[A-Za-z_]')
_SAFE_CHARS = re.compile(r'[^A-Za-z0-9_]')

def to_safe_identifier(col: t.Any) -> str:
    """
    Make a Snowflake-safe identifier from any column label.
    - replace non-word chars with underscores
    - ensure starts with letter/_; then UPPERCASE
    """
    s = str(col).strip()
    s = _SAFE_CHARS.sub('_', s)
    if not _SAFE_START.match(s):
        s = f'_{s}'
    return s.upper()

def sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping = {c: to_safe_identifier(c) for c in df.columns}
    out = df.copy()
    out.columns = [mapping[c] for c in df.columns]
    return out, mapping

def _as_basic_scalar(v: t.Any) -> BasicScalar:
    """Coerce Pandas/Numpy scalars to plain Python types; NaN -> None."""
    if pd.isna(v):
        return None
    if isinstance(v, (str, int, float)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return str(v)

def validate_df_generic(df: pd.DataFrame, sample_rows: int = 200) -> tuple[bool, str]:
    """Light Pydantic check on a small sample of rows."""
    try:
        for rec in df.head(sample_rows).to_dict(orient="records"):
            clean = {k: _as_basic_scalar(v) for k, v in rec.items()}
            AnyRow.model_validate(clean)
        return True, "OK"
    except Exception as e:
        return False, f"Pydantic basic validation failed: {e}"

# ---------------- Snowflake utilities ----------------
def ensure_objects(conn):
    """Ensure schemas exist. Tables are created dynamically by the app."""
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB}.{RAW}")
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DB}.{PROC}")

def get_columns_and_types(conn) -> pd.DataFrame:
    sql = f"""
      SELECT COLUMN_NAME, DATA_TYPE
      FROM {DB}.INFORMATION_SCHEMA.COLUMNS
      WHERE TABLE_SCHEMA = '{RAW}' AND TABLE_NAME = 'STAGING'
      ORDER BY ORDINAL_POSITION
    """
    return pd.read_sql(sql, conn)

def build_missing_summary(conn) -> pd.DataFrame:
    cols = get_columns_and_types(conn)
    if cols.empty:
        return pd.DataFrame(columns=["COLUMN_NAME", "TOTAL_ROWS", "MISSING_COUNT", "MISSING_RATE"])

    cur = conn.cursor()
    total = cur.execute(f"SELECT COUNT(*) FROM {STAGING}").fetchone()[0]
    rows = []
    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        miss = cur.execute(f"SELECT COUNT(*) FROM {STAGING} WHERE {col} IS NULL").fetchone()[0]
        rate = (miss / total) if total else 0
        rows.append([col, total, miss, rate])
    return pd.DataFrame(rows, columns=["COLUMN_NAME", "TOTAL_ROWS", "MISSING_COUNT", "MISSING_RATE"])

def _llm_complete(conn, model: str, prompt: str) -> str:
    """Try AI_COMPLETE first; fall back to SNOWFLAKE.CORTEX.COMPLETE."""
    try:
        df = pd.read_sql(
            "SELECT AI_COMPLETE(model => %s, prompt => %s) AS RESP",
            conn, params=[model, prompt],
        )
        return str(df.iloc[0, 0]).strip()
    except Exception:
        pass
    df = pd.read_sql(
        "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS RESP",
        conn, params=[model, prompt],
    )
    return str(df.iloc[0, 0]).strip()

def _normalize_method(text: t.Optional[str]) -> t.Optional[str]:
    if not text:
        return None
    tkn = str(text).strip().lower()
    for m in ALLOWED_METHODS:
        if m in tkn:
            return m
    return None

def pick_method_ensemble(conn, col_name: str, miss: int, dtype: str) -> tuple[str, str, str]:
    base_prompt = (
        "Pick the best single-word imputation method for a column with missing values. "
        f"Allowed: {', '.join(sorted(ALLOWED_METHODS))}. "
        f"Column={col_name}, Missing={miss}, Type={dtype}. "
        "Return ONLY the one word."
    )
    votes: list[tuple[str, t.Optional[str]]] = []
    for model in [MODEL_A, MODEL_B, MODEL_C]:
        try:
            raw = _llm_complete(conn, model, base_prompt)
            votes.append((model, _normalize_method(raw)))
        except Exception:
            votes.append((model, None))

    counts: dict[str, int] = {}
    for _, v in votes:
        if v:
            counts[v] = counts.get(v, 0) + 1

    if counts:
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if n >= 2:
            return best, f"ensemble-mode:{n}", ", ".join([f"{m}:{v or 'none'}" for m, v in votes])

    # Arbiter
    try:
        vote_str = ", ".join([f"{m}:{v or 'none'}" for m, v in votes])
        arb_prompt = (
            "You are the judge. Choose the single best imputation method. "
            f"Allowed: {', '.join(sorted(ALLOWED_METHODS))}. "
            f"Column={col_name}, Missing={miss}, Type={dtype}. "
            f"Model votes: {vote_str}. Return ONLY the chosen word."
        )
        arb = _normalize_method(_llm_complete(conn, ARBITER_MODEL, arb_prompt))
        if arb:
            return arb, f"arbiter:{ARBITER_MODEL}", vote_str
    except Exception:
        pass

    # Fallback heuristic
    if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
        return "median", "heuristic", ", ".join([f"{m}:{v or 'none'}" for m, v in votes])
    return "mode", "heuristic", ", ".join([f"{m}:{v or 'none'}" for m, v in votes])

def apply_imputations(conn, use_llm: bool) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute(f"CREATE OR REPLACE TABLE {WORKING} AS SELECT * FROM {STAGING}")

    cols = pd.read_sql(
        f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM {DB}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'STAGING'
        ORDER BY ORDINAL_POSITION
        """,
        conn, params=[RAW],
    )

    summary_rows = []
    for _, r in cols.iterrows():
        col = r["COLUMN_NAME"]
        dtype = r["DATA_TYPE"]
        miss = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        if miss == 0:
            summary_rows.append([col, miss, 0.0, "none", "none", ""])
            continue

        if use_llm:
            method, source, votes = pick_method_ensemble(conn, col, miss, dtype)
        else:
            method = "median" if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]) else "mode"
            source = "heuristic"
            votes = ""

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
            cur.execute(f"UPDATE {WORKING} SET {col} = 'UNKNOWN' WHERE {col} IS NULL")

        total_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING}").fetchone()[0]
        missing_after = cur.execute(f"SELECT COUNT(*) FROM {WORKING} WHERE {col} IS NULL").fetchone()[0]
        rate_after = (missing_after / total_after) if total_after else 0.0
        summary_rows.append([col, miss, rate_after, method, source, votes])

    cur.execute(f"CREATE OR REPLACE TABLE {CLEANED} AS SELECT * FROM {WORKING}")

    sdf = pd.DataFrame(
        summary_rows,
        columns=["COLUMN_NAME", "MISSING_BEFORE", "MISSING_RATE_AFTER", "IMPUTATION_METHOD", "SOURCE", "VOTES"],
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

# ---------------- UI ----------------
# Title
st.markdown("<div class='h1-title'>Missile AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI powered app that fills in missing data in a table</div>", unsafe_allow_html=True)

# Watch Tutorial button -> toggle embedded video inline
if "show_tutorial" not in st.session_state:
    st.session_state["show_tutorial"] = False

if st.button("📺 Watch Tutorial", use_container_width=True):
    st.session_state["show_tutorial"] = not st.session_state["show_tutorial"]

if st.session_state["show_tutorial"]:
    st.video("https://youtu.be/YBxv84lqPFs?si=c3rtTZFo_aZ85exH")

conn = get_connector()
ensure_objects(conn)

# ---- Upload CSV (arbitrary columns) ----
st.markdown("<div class='section-title'>Upload CSV</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["csv"])  # empty label; we render our own title
df_uploaded: t.Optional[pd.DataFrame] = None

if uploaded:
    raw_df = pd.read_csv(uploaded)
    df_uploaded, name_map = sanitize_columns(raw_df)

    # Show any renames to the user
    renamed = {orig: safe for orig, safe in name_map.items() if orig != safe}
    if renamed:
        st.info(
            "Some column names were adjusted to be Snowflake-safe identifiers:\n\n" +
            "\n".join([f"- **{o}** → `{s}`" for o, s in renamed.items()])
        )

    st.subheader("Preview:")
    st.dataframe(df_uploaded.head())

    ok, msg = validate_df_generic(df_uploaded)
    if ok:
        st.success("Pydantic (basic) validation: OK")
    else:
        st.error(msg)

    if ok and st.button("Upload to Snowflake (STAGING)", use_container_width=True):
        cur = conn.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {STAGING}")

        session = get_session()
        session.write_pandas(
            df_uploaded,
            table_name="STAGING",
            database=DB,
            schema=RAW,
            auto_create_table=True,
            overwrite=False,
        )
        st.success(f"Uploaded {len(df_uploaded):,} rows into {STAGING}")

st.divider()

# ---- Analyze & Impute ----
st.markdown("<div class='section-title'>Analyze & Impute</div>", unsafe_allow_html=True)
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

st.divider()

# ---- Preview Tables ----
st.markdown("<div class='section-title'>Preview Table and Action Summary</div>", unsafe_allow_html=True)
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

st.divider()

# ---- Finish ----
st.markdown("<div class='section-title'>Finish</div>", unsafe_allow_html=True)
if st.button("Publish (Finalize & Clean)", use_container_width=True):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT 1 FROM {CLEANED} LIMIT 1")
        cur.execute(f"CREATE OR REPLACE TABLE {DB}.{RAW}.FINAL AS SELECT * FROM {CLEANED}")
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
