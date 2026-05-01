WITH source_data AS(
    SELECT
        *
    FROM
        read_parquet('data/raw/parquet/p148_t03_employment.parquet')
)

SELECT
    *,

    CAST(
        ROUND(
            100 *
            employment_thousands /
            FIRST_VALUE(employment_thousands) OVER (ORDER BY year)
        ) AS INTEGER
    ) AS relative_number_1899_100

FROM source_data
