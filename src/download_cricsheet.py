import os
import io
import zipfile
import requests

OUT_DIR = "data/cricsheet_t20i_json"


def main():
    zip_url = os.getenv("T20I_JSON_ZIP_URL")

    if not zip_url:
        raise RuntimeError(
            "T20I_JSON_ZIP_URL not set. "
            "Add it in Replit Secrets from https://cricsheet.org/downloads/")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Downloading Cricsheet T20I JSON...")
    response = requests.get(zip_url, timeout=120)
    response.raise_for_status()

    print("Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(OUT_DIR)

    print(f"Done. Files extracted to {OUT_DIR}")


if __name__ == "__main__":
    main()
