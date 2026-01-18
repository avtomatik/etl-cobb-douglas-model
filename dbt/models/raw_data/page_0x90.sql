WITH raw_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/page_0x90.parquet')
)

SELECT
    period,
    val,
    ROUND(val * 100 / SUM(val) OVER (), 1) AS pct
FROM
    raw_data
