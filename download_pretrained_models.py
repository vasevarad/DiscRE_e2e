import gdown

def download_folder_from_gdrive(folder_url, output_path=None):
    gdown.download_folder(folder_url, output=output_path, quiet=False, use_cookies=False)

def download_file_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def main():
    file_id = "1_p31FxG0ZwPCQlpOj0tbybxx3U8of9Th"
    output_path = "dummy_texts.csv"
    download_file_from_gdrive(file_id, output_path)

    folder_url = "https://drive.google.com/drive/folders/1o8UUyqkR_YfN_PHKESMJKygBdNpOW_TR"
    download_folder_from_gdrive(folder_url)

if __name__ == "__main__":
    main()