SELECT
    year,
    employment_thousands,
    ROUND(
        100 *
        employment_thousands /
        FIRST_VALUE(employment_thousands) OVER (ORDER BY year)
    ) AS relative_number_1899_100
FROM {{ ref('raw_p148_t03_employment') }}
