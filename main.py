import streamlit as st
import pandas as pd
import snowflake.connector

# Snowflake connection
@st.cache_resource
def get_sf():
    return snowflake.connector.connect(
        account=st.secrets["snowflake"]["account"],
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        role=st.secrets["snowflake"]["role"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database="DEMO_DB",
        schema="RAW"
    )

st.title("Missle AI Tool")

conn = get_sf()
cur = conn.cursor()

# Upload CSV directly into Snowflake
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())

    # Write to Snowflake STAGING table
    success, nchunks, nrows, _ = cur.execute_streamlit_upload(df, "STAGING")
    st.success(f"Uploaded {nrows} rows into STAGING")

# Run analysis
if st.button("Analyze Missing Data"):
    cur.execute("CALL DEMO_DB.PROC.ANALYZE();")
    st.success("SUMMARY table updated.")

# Run imputation
if st.button("Impute Missing Data (Cortex + Snowpark)"):
    cur.execute("CALL DEMO_DB.PROC.IMPUTE();")
    st.success("CLEANED + SUMMARY tables updated.")

# Show results
if st.button("Show Tables"):
    summary = cur.execute("SELECT * FROM DEMO_DB.PROC.SUMMARY").fetch_pandas_all()
    cleaned = cur.execute("SELECT * FROM DEMO_DB.PROC.CLEANED LIMIT 50").fetch_pandas_all()
    st.subheader("Summary Table")
    st.dataframe(summary)
    st.subheader("Cleaned Data (sample)")
    st.dataframe(cleaned)

