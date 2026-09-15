from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(name="manual_transactions", comment="Unified Silver transactions from manual bronze")
@dp.expect_or_drop("valid_amount", "amount IS NOT NULL")
@dp.expect_or_drop("valid_date", "date IS NOT NULL")
def manual_transactions():
    return (
        spark.read.table("bronze.manual_raw")
        .select(
            F.regexp_extract("entry_id", r"(\d+)$", 1).cast("int").alias("transaction_id"),
            F.col("user_id").alias("user_id"),
            F.to_date("entry_ts").alias("date"),
            F.col("amount").alias("amount"),
            F.col("currency").alias("currency"),
            F.col("direction").alias("direction"),
            F.col("merchant_raw").alias("merchant_raw"),
            F.col("category_hint").alias("category_hint"),
            F.lit("manual").alias("source"),
            F.col("note"),
        )
        .withColumn("is_card", F.col("note").isNull())
        .withColumn("is_cash", ~F.col("is_card"))
    )
