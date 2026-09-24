from pyspark import pipelines as dp
from pyspark.sql import functions as F

dp.create_streaming_table(
    name="silver.transactions",
    comment="Unified Silver transactions across all sources",
)


@dp.append_flow(
    target="silver.transactions",
    name="manual_transactions_flow",
)
# @dp.expect_or_drop("valid_amount", "amount IS NOT NULL")
# @dp.expect_or_drop("valid_date", "date IS NOT NULL")
def manual_transactions():
    return spark.readStream.table("bronze.manual_raw").select(  # noqa: F821
        F.col("entry_id").alias("transaction_id"),
        F.lit(None).cast("string").alias("account_id"),
        F.col("user_id").alias("user_id"),
        F.to_date("entry_ts").alias("date"),
        (F.col("amount").cast("double") / 100).alias("amount"),
        F.col("currency").alias("currency"),
        F.col("direction").alias("direction"),
        F.col("merchant_raw").alias("merchant_raw"),
        F.lit("manual").alias("source"),
        F.col("category_hint").alias("category_id"),
        F.lit("manual").alias("category_source"),
        F.lit(None).cast("double").alias("confidence"),
    )


@dp.append_flow(
    target="silver.transactions",
    name="mono_transactions_flow",
)
def mono_transactions():
    return spark.readStream.table("bronze.mono_raw").select(  # noqa: F821
        F.col("transaction_id").alias("transaction_id"),
        F.col("account_id").alias("account_id"),
        F.lit("user-oleksandr").alias("user_id"),
        F.to_date(F.timestamp_seconds("time_unix")).alias("date"),
        (F.col("amount").cast("double") / 100).alias("amount"),
        F.when(F.col("currency_code") == 980, "UAH").otherwise(F.col("currency_code").cast("string")).alias("currency"),
        F.when(F.col("amount") < 0, "expense").otherwise("income").alias("direction"),
        F.col("description").alias("merchant_raw"),
        F.lit("monobank").alias("source"),
        F.lit(None).cast("string").alias("category_id"),
        F.lit(None).cast("string").alias("category_source"),
        F.lit(None).cast("double").alias("confidence"),
    )
