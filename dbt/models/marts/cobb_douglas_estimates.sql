SELECT
    regr_slope(
        LN(labor_productivity),
        LN(labor_capital_intensity)
    ) AS alpha,
    EXP(
        regr_intercept(
            LN(labor_productivity),
            LN(labor_capital_intensity)
        )
    ) AS scale
FROM
    {{ ref('int_productivity_metrics') }}
