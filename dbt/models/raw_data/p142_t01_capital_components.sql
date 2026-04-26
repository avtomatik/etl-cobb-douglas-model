WITH raw_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/parquet/p142_t01_capital_components.parquet')
)

SELECT
    period,
    CDT1S1,
    CDT1S2,
    CDT1S3,
    CDT1S4,
    CDT1S3 + CDT1S4 AS CDT1S5
FROM
    raw_data
