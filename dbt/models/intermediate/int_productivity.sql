WITH normalized_data AS (
    SELECT
        period,
        capital AS base_capital,
        labor AS base_labor,
        product AS base_product
    FROM
        {{ ref('int_inputs') }}
    WHERE
        period = {{ var('base_year') }}
)
SELECT
    t.period,
    100 * t.capital / b.base_capital AS capital_norm,
    100 * t.labor / b.base_labor AS labor_norm,
    100 * t.product / b.base_product AS product_norm,
    (100 * t.capital / b.base_capital) / (100 * t.labor / b.base_labor) AS labor_capital_intensity,
    (100 * t.product / b.base_product) / (100 * t.labor / b.base_labor) AS labor_productivity,
    (100 * t.labor / b.base_labor) / (100 * t.capital / b.base_capital) AS capital_labor_ratio,
    (100 * t.product / b.base_product) / (100 * t.capital / b.base_capital) AS capital_turnover
FROM
    {{ ref('int_inputs') }} t
    JOIN normalized_data b ON 1 = 1
