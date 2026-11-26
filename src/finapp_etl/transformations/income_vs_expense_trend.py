from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.table(name="gold.income_vs_expense_trend", comment="Gold aggregation for tracking income/expense diff")
def income_vs_expense_trend():
    return (
        spark.read.table("silver.transactions")  # noqa: F821
        .groupBy(
            "user_id",
            "currency",
            F.trunc("date", "MM").alias("month"),
        )
        .agg(
            F.sum(F.when(F.col("direction") == "income", F.col("amount")).otherwise(0)).alias("income_total"),
            F.sum(F.when(F.col("direction") == "expense", F.col("amount")).otherwise(0)).alias("expense_total"),
        )
        .withColumn("net", F.col("income_total") - F.col("expense_total"))
    )
