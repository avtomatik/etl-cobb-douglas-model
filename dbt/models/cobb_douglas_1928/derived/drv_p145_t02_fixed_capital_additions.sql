WITH base AS (
    SELECT *
    FROM {{ ref('raw_p145_t02_fixed_capital_additions') }}
),

calc AS (
    SELECT
        year,
        annual_increase_cost_price_musd,
        cost_index_1880eq100,
        annual_increase_1880usd_musd,

        SUM(annual_increase_cost_price_musd)
            OVER (ORDER BY year)
            + 4062
            AS total_fixed_capital_1880_dollars

    FROM base
),

base_year AS (
    SELECT total_fixed_capital_1880_dollars AS base_capital
    FROM calc
    WHERE year = 1899
)

SELECT
    c.year,
    c.annual_increase_cost_price_musd,
    c.cost_index_1880eq100,
    c.annual_increase_1880usd_musd,
    c.total_fixed_capital_1880_dollars,

    ROUND(
        100 * c.total_fixed_capital_1880_dollars / b.base_capital
    ) AS relative_total_capital_1899_100

FROM calc c
CROSS JOIN base_year b
ORDER BY year