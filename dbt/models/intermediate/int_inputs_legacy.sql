{% set spec_name = var('active_cobb_douglas_spec') %}
{% set spec = var('cobb_douglas_specs')[spec_name] %}

WITH capital AS (
    SELECT
        year,
        value AS capital
    FROM
        {{ ref(spec.capital.model) }}
    WHERE
        series_id = '{{ spec.capital.series_id }}'
),

labor AS (
    SELECT
        year,
        value AS labor
    FROM
        {{ ref(spec.labor.model) }}
    WHERE
        series_id = '{{ spec.labor.series_id }}'
),

product AS (
    SELECT
        year,
        value AS product
    FROM
        {{ ref(spec.product.model) }}
    WHERE
        series_id = '{{ spec.product.series_id }}'
)

SELECT
    capital.year,
    capital.capital,
    labor.labor,
    product.product
FROM
    capital
    JOIN labor USING (year)
    JOIN product USING (year)
ORDER BY year
