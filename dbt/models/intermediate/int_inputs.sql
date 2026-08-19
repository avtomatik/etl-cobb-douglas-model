{% set spec_name = var('active_cobb_douglas_spec') %}
{% set spec = var('cobb_douglas_specs')[spec_name] %}

{% if spec.capital.column is defined %}

-- Wide-table paper sources

WITH capital AS (
    SELECT
        year,
        {{ spec.capital.column }} AS capital
    FROM {{ ref(spec.capital.model) }}
),

labor AS (
    SELECT
        year,
        {{ spec.labor.column }} AS labor
    FROM {{ ref(spec.labor.model) }}
),

product AS (
    SELECT
        year,
        {{ spec.product.column }} AS product
    FROM {{ ref(spec.product.model) }}
)

SELECT
    c.year,
    c.capital,
    l.labor,
    p.product
FROM capital c
JOIN labor l USING (year)
JOIN product p USING (year)
ORDER BY c.year

{% else %}

-- Long-format JSON series

WITH capital AS (
    SELECT
        year,
        value AS capital
    FROM {{ ref(spec.capital.model) }}
    WHERE series_code = '{{ spec.capital.series_code }}'
),

labor AS (
    SELECT
        year,
        value AS labor
    FROM {{ ref(spec.labor.model) }}
    WHERE series_code = '{{ spec.labor.series_code }}'
),

product AS (
    SELECT
        year,
        value AS product
    FROM {{ ref(spec.product.model) }}
    WHERE series_code = '{{ spec.product.series_code }}'
)

SELECT
    c.year,
    c.capital,
    l.labor,
    p.product
FROM capital c
JOIN labor l USING (year)
JOIN product p USING (year)
ORDER BY c.year

{% endif %}
