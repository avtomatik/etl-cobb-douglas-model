WITH t05 AS (
    SELECT *
    FROM {{ ref('drv_p150_t05_labor_capital_proportions') }}
),

t07 AS (
    SELECT *
    FROM {{ ref('drv_p153_t07_product_deviations_from_trend') }}
),

t08 AS (
    SELECT *
    FROM {{ ref('drv_p159_t08_recorded_vs_calculated_product') }}
),

t10 AS (
    SELECT *
    FROM {{ ref('raw_p163_t10_value_productivity_per_labor') }}
),

t11 AS (
    SELECT *
    FROM {{ ref('raw_p164_t11_productivity_vs_wages') }}
),

t09 AS (
    SELECT
        year,
        value_product_total_index
    FROM {{ ref('raw_p162_t09_value_product_prices') }}
),

t06 AS (
    SELECT
        year,
        product_calculated,
        product_recorded,
        product_pct_deviation
    FROM {{ ref('raw_p152_t06_business_annals') }}
)

SELECT
    -- timeline anchor
    t07.year,

    -- Table V (factor proportions)
    t05.labor_to_capital_index_1899_100,

    -- Table VI (business cycle / production comparison)
    t06.product_recorded,
    t06.product_calculated,
    t06.product_pct_deviation,

    -- Table VII (trend deviations)
    t07.recorded_product_deviation_from_trend,
    t07.calculated_product_deviation_from_trend,

    -- Table VIII (model fit)
    t08.recorded_product_index,
    t08.calculated_product_index,
    t08.calculated_vs_recorded_product_deviation_pct,

    -- Table IX (value product)
    t09.value_product_total_index,

    -- Table X (productivity)
    t10.value_product_per_worker_index,

    -- Table XI (wages vs productivity)
    t11.value_product_per_worker_index,
    t11.real_wages_index,
    t11.real_wages_deviation_from_productivity_pct,
    t11.business_annals_note

FROM t07
LEFT JOIN t05 USING (year)
LEFT JOIN t06 USING (year)
LEFT JOIN t08 USING (year)
LEFT JOIN t09 USING (year)
LEFT JOIN t10 USING (year)
LEFT JOIN t11 USING (year)

ORDER BY t07.year
