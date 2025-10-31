import logging
import csv


def init_logging(debug_path='output.debug'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(debug_path, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def init_leaderboard(file_path='leaderboard.csv'):
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['name', 'accuracy', 'f1-score', 'precision', 'recall'])


def update_leaderboard(result, file_path='leaderboard.csv'):
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result)

