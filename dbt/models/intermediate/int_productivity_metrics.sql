WITH normalized_data AS (
    SELECT
        year,
        capital_norm,
        labor_norm,
        product_norm
    FROM
        {{ ref('int_normalized_data') }}
),
product_trend_and_gap AS (
    SELECT
        year,
        capital_norm,
        labor_norm,
        product_norm,
        -- Calculate derived productivity metrics
        (capital_norm / labor_norm) AS labor_capital_intensity,
        (product_norm / labor_norm) AS labor_productivity,
        (labor_norm / capital_norm) AS capital_labor_ratio,
        (product_norm / capital_norm) AS capital_turnover,
        -- Compute the product trend (3-year rolling average)
        AVG(product_norm) OVER (
            ORDER BY
                year ROWS BETWEEN 1 PRECEDING
                AND 1 FOLLOWING
        ) AS product_trend,
        -- Compute the product gap (deviation from trend)
        product_norm - AVG(product_norm) OVER (
            ORDER BY
                year ROWS BETWEEN 1 PRECEDING
                AND 1 FOLLOWING
        ) AS product_gap
    FROM
        normalized_data
)
SELECT
    *
FROM
    product_trend_and_gap
