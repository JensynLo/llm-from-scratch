import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import argparse

load_dotenv()

project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.data import extract_urls_samples, process_large_file
import concurrent.futures
from scripts.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def download_data(config: dict) -> None:
    urls_file = config["paths"]["urls_path"]
    max_samples = config["data"]["max_samples"]
    output_name = config["output"]["webpage_file_name"]
    output_dir = Path(config["data"]["output_dir"])
    logger.info(f"Starting parallel data extraction from {str(urls_file)}...")
    output_path = output_dir / output_name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        future_wiki = executor.submit(
            extract_urls_samples,
            str(urls_file),
            str(output_dir / output_name),
            max_samples,
        )

        try:
            future_wiki.result()
            logger.info("Data extraction completed!")
        except Exception as e:
            logger.error(f"Error during data extraction: {e}")
            raise


def clean_data(config: dict) -> None:
    output_name = config["output"]["webpage_file_name"]
    cleaned_output_name = config["output"]["webpage_cleaned_file_name"]
    output_dir = Path(config["data"]["output_dir"])
    chunk_size = config["clean"]["chunk_size"]
    output_path = output_dir / cleaned_output_name

    logger.info(f"Starting parallel data cleaning (chunk_size={chunk_size})...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        future = executor.submit(
            process_large_file,
            input_file=str(output_dir / output_name),
            output_file=str(output_path),
            chunk_size=chunk_size,
        )

        try:
            future.result()
            logger.info("Data cleaning completed!")
        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")
            raise


def run(
    config_path: str,
    skip_download: bool = False,
    skip_clean: bool = False,
):
    setup_logging()

    config = load_config(config_path)

    if not skip_download:
        download_data(config)
    else:
        logger.info("Skipping data download")

    if not skip_clean:
        clean_data(config)
    else:
        logger.info("Skipping data cleaning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data pipeline runner")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip data download"
    )
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning")

    args = parser.parse_args()
    run(args.config, args.skip_download, args.skip_clean)
