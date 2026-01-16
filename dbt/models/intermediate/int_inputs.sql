{% set spec_name = var('active_cobb_douglas_spec') %}
{% set spec = var('cobb_douglas_specs')[spec_name] %}
{% set base_year = spec.base_year %}

WITH capital AS (
    SELECT
        period,
        value AS capital
    FROM
        {{ ref(spec.capital.model) }}
    WHERE
        series_id = '{{ spec.capital.series_id }}'
),

labor AS (
    SELECT
        period,
        value AS labor
    FROM
        {{ ref(spec.labor.model) }}
    WHERE
        series_id = '{{ spec.labor.series_id }}'
),

product AS (
    SELECT
        period,
        value AS product
    FROM
        {{ ref(spec.product.model) }}
    WHERE
        series_id = '{{ spec.product.series_id }}'
)

SELECT
    capital.period,
    capital.capital,
    labor.labor,
    product.product
FROM
    capital
    JOIN labor USING (period)
    JOIN product USING (period)
