WITH source_data AS (
    SELECT
        *
    FROM
        read_parquet('data/raw/parquet/p144_specified_capital_goods.parquet')
)

SELECT
    *,

    ROUND(
        specified_capital_goods_value_musd *
        100 / SUM(specified_capital_goods_value_musd) OVER (),
        1
    ) AS specified_capital_goods_pct

FROM source_data
