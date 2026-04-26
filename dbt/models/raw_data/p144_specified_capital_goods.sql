WITH raw_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/parquet/p144_specified_capital_goods.parquet')
)

SELECT
    period,
    val,
    ROUND(val * 100 / SUM(val) OVER (), 1) AS pct
FROM
    raw_data
