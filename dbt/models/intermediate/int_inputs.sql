WITH capital AS (
    SELECT
        period,
        value AS capital
    FROM
        {{ ref('stg_usa_cobb_douglas') }}
    WHERE
        series_id = 'CDT2S4'
),
labor AS (
    SELECT
        period,
        value AS labor
    FROM
        {{ ref('stg_usa_cobb_douglas') }}
    WHERE
        series_id = 'CDT3S1'
),
product AS (
    SELECT
        period,
        value AS product
    FROM
        {{ ref('stg_uscb') }}
    WHERE
        series_id = 'J0014'
)
SELECT
    capital.period,
    capital,
    labor,
    product
FROM
    capital
    JOIN labor USING (period)
    JOIN product USING (period)
