WITH regression AS (
    SELECT
        *
    FROM
        {{ ref('cobb_douglas_estimates') }}
),
product_model_data AS (
    SELECT
        i.*,
        e.alpha,
        e.scale,
        e.scale * POWER(i.capital_norm, e.alpha) * POWER(i.labor_norm, 1 - e.alpha) AS product_model
    FROM
        {{ ref('int_productivity_metrics') }} i
        CROSS JOIN regression e
)
SELECT
    pmd.period,
    pmd.capital_norm,
    pmd.labor_norm,
    pmd.product_norm,
    pmd.labor_capital_intensity,
    pmd.labor_productivity,
    pmd.capital_labor_ratio,
    pmd.capital_turnover,
    pmd.product_trend,
    pmd.product_gap,
    pmd.product_model,
    -- Calculate the product model trend (3-period rolling average)
    AVG(pmd.product_model) OVER (
        ORDER BY
            pmd.period ROWS BETWEEN 1 PRECEDING
            AND 1 FOLLOWING
    ) AS product_model_trend,
    -- Calculate the product model gap (deviation from trend)
    pmd.product_model - AVG(pmd.product_model) OVER (
        ORDER BY
            pmd.period ROWS BETWEEN 1 PRECEDING
            AND 1 FOLLOWING
    ) AS product_model_gap,
    -- Calculate the product model error (deviation from actual product)
    pmd.product_model / pmd.product_norm - 1 AS product_model_error
FROM
    product_model_data pmd
