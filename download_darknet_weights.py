""" """
import os
import subprocess
import time
import sys
import argparse
import requests
import progressbar


def parse_args():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument(
        "--download_folder_path", type=str, default="",
        help="Folder to download weights to."
    )
    parser.add_argument(
        "--tiny", type=bool, default=True,
        help="True for tiny-yolov3 and False for yolov3."
    )
    parser.add_argument(
        "--config_file_path", type=str, default="",
        help="Path to darknet config file."
    )

    return parser.parse_args()


def download_darknet_weights(url, download_folder_path):
    r = requests.get(url, stream=True)

    f = open(os.path.join(download_folder_path, url.split("/")[-1]), 'wb')
    file_size = int(r.headers.get('content-length'))
    chunk = 100
    num_bars = file_size // chunk
    bar = progressbar.ProgressBar(maxval=num_bars).start()
    i = 0
    for chunk in r.iter_content(chunk):
        f.write(chunk)
        bar.update(i)
        i += 1
    f.close()


if __name__ == '__main__':
    args = parse_args()

    if args.tiny:
        url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
    else:
        url = 'https://pjreddie.com/media/files/yolov3.weights'

    # Download darknet weights
    download_darknet_weights(url, args.download_folder_path)
