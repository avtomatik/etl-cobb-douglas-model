WITH labor AS (
    SELECT
        year,
        employment_index_1899_100 AS labor_index
    FROM {{ ref('raw_p148_t03_employment') }}
),

capital AS (
    SELECT
        year,
        capital_index_1899_100 AS capital_index
    FROM {{ ref('raw_p145_t02_fixed_capital_additions') }}
),

recorded AS (
    SELECT
        year,
        manufacturing_output_index AS recorded_product
    FROM {{ ref('raw_p149_t04_manufacturing_output_index') }}
),

base AS (
    SELECT
        r.year,
        r.recorded_product,

        1.01
        * POWER(l.labor_index, 0.75)
        * POWER(c.capital_index, 0.25) AS calculated_product

    FROM recorded r
    JOIN labor l USING (year)
    JOIN capital c USING (year)
),

trend AS (
    SELECT
        *,
        AVG(recorded_product) OVER (
            ORDER BY year
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ) AS recorded_trend,

        AVG(calculated_product) OVER (
            ORDER BY year
            ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        ) AS calculated_trend

    FROM base
)

SELECT
    year,

    ROUND(
        recorded_product - recorded_trend,
        1
    ) AS recorded_product_deviation_from_trend,

    ROUND(
        calculated_product - calculated_trend,
        1
    ) AS calculated_product_deviation_from_trend

FROM trend
WHERE year NOT IN (
    (SELECT MIN(year) FROM trend),
    (SELECT MAX(year) FROM trend)
)
ORDER BY year
