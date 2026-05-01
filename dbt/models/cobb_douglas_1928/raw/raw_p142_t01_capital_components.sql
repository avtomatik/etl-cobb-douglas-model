WITH source_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/parquet/p142_t01_capital_components.parquet')
)

SELECT
    *,
    buildings_value_musd + machinery_value_musd AS capital_total_musd
FROM source_data
