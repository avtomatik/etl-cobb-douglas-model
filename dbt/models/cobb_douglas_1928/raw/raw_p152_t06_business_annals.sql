WITH labor AS (
    SELECT year, relative_number_1899_100
    FROM {{ ref('raw_p148_t03_employment') }}
),
capital AS (
    SELECT year, relative_total_capital_1899_100
    FROM {{ ref('raw_p145_t02_fixed_capital_additions') }}
),
product_recorded AS (
    SELECT year, manufacturing_output_index
    FROM {{ ref('raw_p149_t04_manufacturing_output_index') }}
),
business_annals AS (
    SELECT *
    FROM read_parquet('data/raw/parquet/p152_t06_business_annals.parquet')
),
calculated AS (
    SELECT
        b.year,
        1.01 * POWER(l.relative_number_1899_100, 0.75) * POWER(c.relative_total_capital_1899_100, 0.25) AS product_calculated_float,
        p.manufacturing_output_index AS product_recorded,
        b.business_annals_note
    FROM
        business_annals b
        JOIN labor l USING (year)
        JOIN capital c USING (year)
        JOIN product_recorded p USING (year)
)

SELECT
    year,
    CAST(product_calculated_float AS INTEGER) AS product_calculated,
    product_recorded,
    ROUND(100 * (1 - product_calculated_float / product_recorded), 1) AS product_pct_deviation,
    business_annals_note
FROM calculated
ORDER BY year
