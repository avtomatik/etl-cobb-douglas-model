{% set source_name = var('active_input_source', 'legacy') %}

{% if source_name == 'legacy' %}

SELECT
    *
FROM
    {{ ref('int_inputs_legacy') }}

{% elif source_name == 'original' %}

SELECT
    *
FROM
    {{ ref('int_inputs_original') }}

{% else %}

SELECT
    *
FROM
    {{ ref('int_inputs_legacy') }}

{% endif %}
