WITH productivity AS (
    SELECT
        year,
        value_product_per_worker_index
    FROM {{ ref('raw_p163_t10_value_productivity_per_labor') }}
),

wages AS (
    SELECT
        year,
        real_wages_index,
        business_annals_note
    FROM read_parquet('data/raw/parquet/p164_t11_productivity_vs_wages.parquet')
)

SELECT
    p.year,
    p.value_product_per_worker_index,
    w.real_wages_index,
    -- Per Cent Deviation of Wages from Productivity
    ROUND(
        100.0 * (w.real_wages_index - p.value_product_per_worker_index) 
               / p.value_product_per_worker_index,
        1
    ) AS real_wages_deviation_from_productivity_pct,
    w.business_annals_note

FROM productivity p
JOIN wages w USING (year)
ORDER BY p.year
