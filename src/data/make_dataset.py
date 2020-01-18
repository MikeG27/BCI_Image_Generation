from google_drive_downloader import GoogleDriveDownloader as gdd
import config
import os

def download_file(file_id,dest_path,unzip=False):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=dest_path,
                                        unzip=unzip,showsize=True,overwrite=True)
    print("/ Data was downloaded ")


if __name__ == "__main__":
    file_id = '1nvSTPTUv5r6bc7axUJcz6dKHUjsqdgjm'
    dest_path = config.RAW_EEG_DIR
    filepath=os.path.join(dest_path,"mnist-64s.csv")
    download_file(file_id,filepath)
