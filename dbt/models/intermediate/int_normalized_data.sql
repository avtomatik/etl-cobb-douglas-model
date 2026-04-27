{% set spec = var('cobb_douglas_specs')[var('active_cobb_douglas_spec')] %}
{% set base_year = spec.base_year %}

WITH base AS (
    SELECT
        capital AS base_capital,
        labor AS base_labor,
        product AS base_product
    FROM
        {{ ref('int_inputs') }}
    WHERE
        year = {{ base_year }}
)
SELECT
    t.year,
    100 * t.capital / b.base_capital AS capital_norm,
    100 * t.labor / b.base_labor AS labor_norm,
    100 * t.product / b.base_product AS product_norm
FROM
    {{ ref('int_inputs') }} t
CROSS JOIN base b
ORDER BY t.year
