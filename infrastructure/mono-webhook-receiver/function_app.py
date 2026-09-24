import logging
import os
import uuid
from datetime import UTC, datetime

import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

app = func.FunctionApp()

STORAGE_ACCOUNT_URL = os.environ["BRONZE_STORAGE_ACCOUNT_URL"]
LANDING_FILE_SYSTEM = os.environ.get("BRONZE_LANDING_FILE_SYSTEM", "dev")
LANDING_PATH_PREFIX = os.environ.get("BRONZE_LANDING_PATH_PREFIX", "bronze-landing/mono_webhook")
WEBHOOK_SECRET = os.environ["MONO_WEBHOOK_SECRET"]

_service_client = DataLakeServiceClient(account_url=STORAGE_ACCOUNT_URL, credential=DefaultAzureCredential())


@app.route(route="webhook/mono/{secret}", methods=["POST", "GET"], auth_level=func.AuthLevel.ANONYMOUS)
def mono_webhook(req: func.HttpRequest) -> func.HttpResponse:
    if req.route_params.get("secret") != WEBHOOK_SECRET:
        logging.warning("mono_webhook: invalid path secret")
        return func.HttpResponse(status_code=404)

    if req.method == "GET":
        return func.HttpResponse(status_code=200)

    body = req.get_body()
    if not body:
        return func.HttpResponse(status_code=400)

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    file_path = f"{LANDING_PATH_PREFIX}/{ts}-{uuid.uuid4()}.json"
    fs_client = _service_client.get_file_system_client(LANDING_FILE_SYSTEM)
    file_client = fs_client.create_file(file_path)
    file_client.append_data(body, offset=0, length=len(body))
    file_client.flush_data(len(body))
    return func.HttpResponse(status_code=200)
