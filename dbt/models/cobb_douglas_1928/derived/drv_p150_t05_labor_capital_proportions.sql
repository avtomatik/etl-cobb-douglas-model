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
)

SELECT
    labor.year,
    CAST(
        ROUND(
            100 * labor.labor_index / capital.capital_index,
            1
        ) AS INTEGER
    ) AS relative_labor_to_capital_1899_100
FROM labor
JOIN capital USING (year)
ORDER BY year
