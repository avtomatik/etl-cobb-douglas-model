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
)

SELECT
    labor.year,
    CAST(
        ROUND(
            100 * labor.labor_index / capital.capital_index,
            1
        ) AS INTEGER
    ) AS labor_to_capital_index_1899_100
FROM labor
JOIN capital USING (year)
ORDER BY year
