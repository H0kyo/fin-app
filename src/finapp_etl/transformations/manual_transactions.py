from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.append_flow(
    target="silver.transactions",
    name="manual_transactions_flow",
    comment="Unified Silver transactions from manual bronze",
)
@dp.expect_or_drop("valid_amount", "amount IS NOT NULL")
@dp.expect_or_drop("valid_date", "date IS NOT NULL")
def manual_transactions():
    return spark.readStream.table("bronze.manual_raw").select(  # noqa: F821
        F.regexp_extract("entry_id", r"(\d+)$", 1).cast("int").alias("transaction_id"),
        F.lit(None).cast("string").alias("account_id"),
        F.col("user_id").alias("user_id"),
        F.to_date("entry_ts").alias("date"),
        F.col("amount").alias("amount"),
        F.col("currency").alias("currency"),
        F.col("direction").alias("direction"),
        F.col("merchant_raw").alias("merchant_raw"),
        F.lit("manual").alias("source"),
        F.col("category_hint").alias("category_id"),
        F.lit("manual").alias("category_source"),
        F.lit(None).cast("double").alias("confidence"),
    )
