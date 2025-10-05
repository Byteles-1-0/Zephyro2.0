# script_download_wildfires.py
import requests
import time
from datetime import datetime

API_KEY = 'dfd34078da289f8007167d5804622712'
SOURCE = 'VIIRS_SNPP_NRT'
AREA = '-168,7,-52,84'  # bounding box Nord America
DAY_RANGE = 1
SAVE_PATH = 'wildfires_latest.csv'

URL = f'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}'

def download_wildfires():
    print(f'[{datetime.now()}] Inizio download incendi...')
    try:
        response = requests.get(URL)
        response.raise_for_status()
        with open(SAVE_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f'[{datetime.now()}] Dati incendi salvati in {SAVE_PATH}')
    except requests.RequestException as e:
        print(f'Errore durante il download: {e}')

if __name__ == '__main__':
    download_wildfires()  # esegui subito il primo download
    while True:
        time.sleep(30 * 60)  # attendi 30 minuti
        download_wildfires()
