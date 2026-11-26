from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

_WEBHOOK_SCHEMA = StructType(
    [
        StructField("type", StringType()),
        StructField(
            "data",
            StructType(
                [
                    StructField("account", StringType()),
                    StructField(
                        "statementItem",
                        StructType(
                            [
                                StructField("id", StringType()),
                                StructField("time", LongType()),
                                StructField("description", StringType()),
                                StructField("mcc", IntegerType()),
                                StructField("originalMcc", IntegerType()),
                                StructField("amount", LongType()),
                                StructField("operationAmount", LongType()),
                                StructField("currencyCode", IntegerType()),
                                StructField("commissionRate", LongType()),
                                StructField("cashbackAmount", LongType()),
                                StructField("balance", LongType()),
                                StructField("comment", StringType()),
                                StructField("receiptId", StringType()),
                                StructField("counterEdrpou", StringType()),
                                StructField("counterIban", StringType()),
                                StructField("counterName", StringType()),
                            ]
                        ),
                    ),
                ]
            ),
        ),
    ]
)


@dp.table(name="bronze.mono_raw", comment="Raw Monobank statement items via webhook/backfill")
@dp.expect_or_drop("valid_amount", "amount IS NOT NULL")
@dp.expect_or_drop("valid_transaction_id", "transaction_id IS NOT NULL")
def bronze_mono_raw():
    landing_path = spark.conf.get("mono_webhook_landing_path")  # noqa: F821
    item = F.col("data.statementItem")
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "json")
        .schema(_WEBHOOK_SCHEMA)
        .load(landing_path)
        .where("type = 'StatementItem'")
        .select(
            F.col("data.account").alias("account_id"),
            item["id"].alias("transaction_id"),
            item["time"].alias("time_unix"),
            item["description"].alias("description"),
            item["mcc"].alias("mcc"),
            item["originalMcc"].alias("original_mcc"),
            item["amount"].alias("amount"),
            item["operationAmount"].alias("operation_amount"),
            item["currencyCode"].alias("currency_code"),
            item["commissionRate"].alias("commission_rate"),
            item["cashbackAmount"].alias("cashback_amount"),
            item["balance"].alias("balance"),
            item["comment"].alias("comment"),
            item["receiptId"].alias("receipt_id"),
            item["counterEdrpou"].alias("counter_edrpou"),
            item["counterIban"].alias("counter_iban"),
            item["counterName"].alias("counter_name"),
            F.current_timestamp().alias("_ingested_at"),
            F.lit("monobank_webhook").alias("_source"),
        )
    )
