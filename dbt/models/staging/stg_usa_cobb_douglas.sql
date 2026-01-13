SELECT
    period :: int AS period,
    series_id,
    value :: double AS value
FROM
    {{ source('raw', 'usa_cobb_douglas') }}
