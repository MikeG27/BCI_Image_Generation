import argparse
import os
import sys
from google_drive_downloader import GoogleDriveDownloader as gdd

sys.path.append(os.getcwd())
import config


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gdrive', type=str, default=config.GDRIVE_ID,
                        help='gdrive link with csv file')
    parser.add_argument('-file_name', type=str, default=config.GDRIVE_FILE,
                        help="csv file name ")
    parser.add_argument('-output', type=str, default=config.RAW_EEG_DIR,
                        help="output directory")

    args = parser.parse_args()

    return args


def download_file(file_id, dest_path, unzip=False):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest_path,
                                        unzip=unzip, showsize=True, overwrite=True)
    print("/ Data was downloaded ")


if __name__ == "__main__":

    args = parser()

    file_id = args.gdrive
    file_name = args.file_name
    dest_path = args.output
    filepath = os.path.join(dest_path, file_name)

    download_file(file_id, filepath)
