from pyspark import pipelines as dp

dp.create_streaming_table(
    name="silver.transactions",
    comment="Unified Silver transactions across all sources",
)
