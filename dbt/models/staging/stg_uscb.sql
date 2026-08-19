SELECT
    year,
    series_code,
    value
FROM
    {{ source('raw', 'uscb') }}
