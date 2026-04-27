SELECT
    period :: int AS year,
    series_id,
    value :: double AS value
FROM
    {{ source('raw', 'uscb') }}
