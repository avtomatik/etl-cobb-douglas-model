WITH source_data AS (
    SELECT *
    FROM read_parquet('data/raw/parquet/p162_t09_value_product_prices.parquet')
),
mfg_output AS (
    SELECT year, manufacturing_output_index
    FROM {{ ref('raw_p149_t04_manufacturing_output_index') }}
),
calculated AS (
    SELECT
        s.*,
        CAST(ROUND(100 * price_mfg_index / price_all_commodities_index) AS INTEGER) AS ratio_mfg_share_all_commodities,
        m.manufacturing_output_index
    FROM source_data s
    JOIN mfg_output m USING (year)
)

SELECT
    year,
    price_mfg_index,
    price_all_commodities_index,
    ratio_mfg_share_all_commodities,

    -- "Total Value Product" scaled as 100-based integer
    CAST(
        ROUND(manufacturing_output_index * ratio_mfg_share_all_commodities / 100)
        AS INTEGER
    ) AS total_value_product
FROM calculated
ORDER BY year
