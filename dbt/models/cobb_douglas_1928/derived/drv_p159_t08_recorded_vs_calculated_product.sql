WITH labor AS (
    SELECT
        year,
        relative_number_1899_100 AS labor_index
    FROM {{ ref('raw_p148_t03_employment') }}
),

capital AS (
    SELECT
        year,
        relative_total_capital_1899_100 AS capital_index
    FROM {{ ref('raw_p145_t02_fixed_capital_additions') }}
),

recorded AS (
    SELECT
        year,
        manufacturing_output_index AS recorded_product_index
    FROM {{ ref('raw_p149_t04_manufacturing_output_index') }}
)

SELECT
    r.year,

    r.recorded_product_index,

    CAST(
        ROUND(
        1.01
        * POWER(l.labor_index, 2.0 / 3.0)
        * POWER(c.capital_index, 1.0 / 3.0),
        1
        ) AS INTEGER
    ) AS calculated_product_index,

    ROUND(
        100.0 * (
            1 - (
                (
                    1.01
                    * POWER(l.labor_index, 2.0 / 3.0)
                    * POWER(c.capital_index, 1.0 / 3.0)
                ) / r.recorded_product_index
            )
        ),
        1
    ) AS relative_product_error_pct

FROM recorded r
JOIN labor l USING (year)
JOIN capital c USING (year)
ORDER BY year
