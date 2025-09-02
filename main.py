# main.py
# Streamlit + Snowflake demo
# - Upload CSV
# - Validate with Pydantic
# - Write to Snowflake
# - Analyze nulls & (optionally) use Snowflake Cortex (with LLM ensemble + arbiter)
# - Apply imputations, preview results, and publish

import typing as t
import pandas as pd
import streamlit as st
from pydantic import BaseModel, field_validator

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from snowflake.snowpark import Session

# ────────────────────────────────────────────────────────────────────────────────
# App config
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Missile AI", layout="centered")

# ONE strong CSS block (kept late in the file earlier; here it's fine at top because
# we don't add other style blocks later). If you add more CSS later, put *this* block last.
st.markdown("""
<style>
/* Centered, compressed app container */
[data-testid="block-container"]{
  max-width: 960px !important;        /* page width; tweak if you want narrower/wider */
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

/* Main title (H1) */
h1, .stMarkdown h1{
  text-align: center;
  font-size: 72px;                    /* big hero title */
  line-height: 1.06;
  margin: 0.2em 0 48px;               /* extra space under the title */
  letter-spacing: 0.2px;
  font-kerning: normal;
  font-feature-settings: "kern";
}

/* Section titles */
.section-title{
  text-align:center;
  font-weight: 900;
  font-size: clamp(24px, 3.0vw, 34px); /* smaller than H1, responsive */
  line-height: 1.12;
  letter-spacing: .2px;
  margin: .5rem 0 .5rem;
}

/* Make ALL Streamlit buttons larger (primary/secondary/link/download) */
.stButton > button,
.stDownloadButton > button,
.stLinkButton > button,
button[kind="primary"],
button[kind="secondary"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-secondary"] {
  font-size: 62px !important;         /* TEXT SIZE */
  font-weight: 700 !important;
  padding: 14px 22px !important;      /* BUTTON SIZE */
  min-height: 54px !important;
  border-radius: 12px !important;
}

/* Make the 'Browse files' button in the uploader bigger too */
[data-testid="stFileUploader"] button {
  font-size: 18px !important;
  padding: 10px 18px !important;
  min-height: 44px !important;
  border-radius: 10px !important;
}

/* Slightly bigger subheaders used in previews */
h3, .stMarkdown h3{
  font-size: 28px;
  line-height: 1.2;
}

/* Optional: nicer hr spacing */
hr { margin: 1.25rem 0; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────────
# Constants (DB/Schema/Tables)
# ────────────────────────────────────────────────────────────────────────────────
DB   = "DEMO_DB"
RAW  = "RAW"
PROC = "PROC"

STAGING = f"{DB}.{RAW}.STAGING"
WORKING = f"{DB}.{PROC}.WORKING"
SUMMARY = f"{DB}.{PROC}.SUMMARY"
CLEANED = f"{DB}.{PROC}.CLEANED"

# ────────────────────────────────────────────────────────────────────────────────
# LLM ensemble settings (Snowflake Cortex)
# ────────────────────────────────────────────────────────────────────────────────
ALLOWED_METHODS = {"mean", "median", "mode", "constant", "drop"}
_models = st.secrets.get("cortex_models", {})
MODEL_A = _models.get("model_a", "mistral-large")   # LLM 1
MODEL_B = _models.get("model_b", "llama3-70b")      # LLM 2
MODEL_C = _models.get("model_c", "reka-flash")      # LLM 3
ARBITER_MODEL = _models.get("arbiter", "mistral-large")  # LLM 4 (judge)

# ────────────────────────────────────────────────────────────────────────────────
# Connections
# ────────────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────────────
# Validation model
# ────────────────────────────────────────────────────────────────────────────────
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
    required_cols = {"ID", "COL_A", "COL_B"}
    if set(map(str.upper, df.columns)) != required_cols:
        return False, f"CSV columns must be exactly {sorted(required_cols)} (case-insensitive)."
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]
    try:
        for rec in df.head(200).to_dict(orient="records"):
            UploadRow(**rec)
    except Exception as e:
        return False, f"Pydantic validation failed: {e}"
    return True, "OK"

# ────────────────────────────────────────────────────────────────────────────────
# Snowflake objects
# ────────────────────────────────────────────────────────────────────────────────
def ensure_objects(conn):
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB}.{RAW}.STAGING (
            ID NUMBER,
            COL_A STRING,
            COL_B FLOAT,
            _LOAD_TS TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)

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

# ────────────────────────────────────────────────────────────────────────────────
# Cortex helpers (LLM ensemble)
# ────────────────────────────────────────────────────────────────────────────────
def _llm_complete(conn, model: str, prompt: str) -> str:
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

    vote_str = ", ".join([f"{m}:{v or 'none'}" for m, v in votes])
    try:
        arb_raw = _llm_complete(
            conn,
            ARBITER_MODEL,
            "You are the judge. Choose the single best imputation method. "
            f"Allowed: {', '.join(sorted(ALLOWED_METHODS))}. "
            f"Column={col_name}, Missing={miss}, Type={dtype}. "
            f"Model votes: {vote_str}. Return ONLY the chosen method word."
        )
        arb = _normalize_method(arb_raw)
        if arb:
            return arb, f"arbiter:{ARBITER_MODEL}", vote_str
    except Exception:
        pass

    if any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
        return "median", "heuristic", vote_str
    return "mode", "heuristic", vote_str

# ────────────────────────────────────────────────────────────────────────────────
# Imputation pipeline
# ────────────────────────────────────────────────────────────────────────────────
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
            source, votes = "heuristic", ""

        if method == "drop":
            cur.execute(f"DELETE FROM {WORKING} WHERE {col} IS NULL")
        elif method in {"mean", "median"} and any(x in dtype for x in ["NUMBER", "FLOAT", "INT", "DECIMAL", "DOUBLE"]):
            agg = "AVG" if method == "mean" else "MEDIAN"
            val = cur.execute(f"SELECT {agg}({col}) FROM {WORKING} WHERE {col} IS NOT NULL").fetchone()[0]
            cur.execute(f"UPDATE {WORKING} SET {col} = %s WHERE {col} IS NULL", (val,))
        elif method == "mode":
            res = cur.execute(
                f"""
                SELECT {col} FROM {WORKING}
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

# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("# Missile AI")  # H1 (styled by CSS)

conn = get_connector()
ensure_objects(conn)

# Upload
st.markdown("<div class='section-title'>Upload CSV</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["csv"])
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
        df_to_write = df_uploaded.copy()
        df_to_write.columns = [c.upper() for c in df_to_write.columns]
        session = get_session()
        session.write_pandas(
            df_to_write, table_name="STAGING",
            database=DB, schema=RAW,
            auto_create_table=True, overwrite=False,
        )
        st.success(f"Uploaded {len(df_to_write):,} rows into {STAGING}")

st.divider()

# Analyze & Impute
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

# Preview tables
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

# Finish
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


