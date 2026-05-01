WITH capital AS (
    SELECT
        year,
        total_fixed_capital_1880_dollars AS capital
    FROM {{ ref('raw_p145_t02_fixed_capital_additions') }}
),

labor AS (
    SELECT
        year,
        employment_thousands AS labor
    FROM {{ ref('raw_p148_t03_employment') }}
),

product AS (
    SELECT
        year,
        manufacturing_output_index AS product
    FROM {{ ref('raw_p149_t04_manufacturing_output_index') }}
)

SELECT
    capital.year,
    capital.capital,
    labor.labor,
    product.product
FROM
    capital
    JOIN labor USING (year)
    JOIN product USING (year)
ORDER BY year
