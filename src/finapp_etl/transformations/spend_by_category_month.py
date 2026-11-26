from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(name="gold.spend_by_category_month", comment="Gold aggregation for spent money by categories per month")
def spend_by_category_month():
    return (
        spark.read.table("silver.transactions")  # noqa: F821
        .filter(F.col("direction") == "expense")
        .groupBy(
            "user_id",
            "category_id",
            "currency",
            F.trunc("date", "MM").alias("month"),
        )
        .agg(
            F.sum("amount").alias("total_spend"),
            F.count(F.lit(1)).alias("transaction_count"),
        )
    )
