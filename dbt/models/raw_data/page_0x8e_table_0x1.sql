WITH raw_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/page_0x8e_table_0x1.parquet')
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
