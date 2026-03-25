"""
ingest.py – Data ingestion from Azure Blob Storage.
Falls back to the local CSV if Azure credentials are not configured.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import BLOB_CONN_STR, BLOB_CONTAINER, BLOB_FILE, DATA_PATH


def download_from_blob(dest_path: str = DATA_PATH) -> str:
    """
    Download raw CSV from Azure Blob Storage to *dest_path*.
    Returns the path to the downloaded file.
    """
    if not BLOB_CONN_STR:
        print("[ingest] No AZURE_STORAGE_CONNECTION_STRING found – skipping Blob download.")
        return dest_path

    from azure.storage.blob import BlobServiceClient

    print(f"[ingest] Connecting to Azure Blob Storage (container={BLOB_CONTAINER}, blob={BLOB_FILE}) …")
    client = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
    blob   = client.get_blob_client(container=BLOB_CONTAINER, blob=BLOB_FILE)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(blob.download_blob().readall())

    print(f"[ingest] Downloaded '{BLOB_FILE}' → '{dest_path}'")
    return dest_path


if __name__ == "__main__":
    download_from_blob()
