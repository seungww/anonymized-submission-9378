import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from vrscanner.loader.data_loader import load_data
from vrscanner.loader.batch_loader import load_batch_for_longitudinal_evaluation
from vrscanner.core.model_selection import get_model
from vrscanner.core.trainer import train
from vrscanner.core.evaluator import evaluate
from vrscanner.utils import update_leaderboard


def longitudinal_evaluation(args):
    logging.info(args)
    all_data, labels, class_names, timestamps, _ = load_data(args.path, args.pktcount)

    args.name = _build_name(args)
    norms = args.norm
    batch_size = args.batch_size

    datetimes = pd.to_datetime(timestamps, unit='s')
    date_ranges = {
        'set1': (datetime(2024, 4, 20), datetime(2024, 5, 18, 23, 59, 59)),
        'set2': (datetime(2024, 5, 19), datetime(2024, 6, 30, 23, 59, 59)),
        'set3': (datetime(2024, 7, 1), datetime(2024, 7, 31, 23, 59, 59)),
        'set4': (datetime(2024, 8, 1), datetime(2024, 9, 30, 23, 59, 59)),
    }

    train_data, val_data = [], []
    test_data_1, test_data_2, test_data_3 = [], [], []

    for data in all_data:
        mask_set1 = (datetimes >= date_ranges["set1"][0]) & (datetimes <= date_ranges["set1"][1])
        mask_set2 = (datetimes >= date_ranges["set2"][0]) & (datetimes <= date_ranges["set2"][1])
        mask_set3 = (datetimes >= date_ranges["set3"][0]) & (datetimes <= date_ranges["set3"][1])
        mask_set4 = (datetimes >= date_ranges["set4"][0]) & (datetimes <= date_ranges["set4"][1])

        x_set1 = data[mask_set1]
        x_set2 = data[mask_set2]
        x_set3 = data[mask_set3]
        x_set4 = data[mask_set4]

        y_set1 = labels[mask_set1]
        y_test_1 = labels[mask_set2]
        y_test_2 = labels[mask_set3]
        y_test_3 = labels[mask_set4]

        x_train, x_val, y_train, y_val = train_test_split(
            x_set1, y_set1, test_size=0.2, random_state=42
        )

        train_data.append(x_train)
        val_data.append(x_val)
        test_data_1.append(x_set2)
        test_data_2.append(x_set3)
        test_data_3.append(x_set4)

    train_loader, val_loader, test_1_loader, test_2_loader, test_3_loader, scalers, token_dicts = load_batch_for_longitudinal_evaluation(
        norms, (train_data, y_train), (val_data, y_val),
        (test_data_1, y_test_1), (test_data_2, y_test_2), (test_data_3, y_test_3),
        batch_size
    )

    num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) for i in range(len(norms))]
    classifier = get_model(args, len(class_names), num_tokens_list)
    logging.info(f"Classifier: {classifier}")
    logging.info("Training...")
    classifier = train(args, classifier, train_loader, val_loader)

    for i, loader in enumerate([test_1_loader, test_2_loader, test_3_loader], start=1):
        test_accuracy, test_f1, test_precision, test_recall = evaluate(args, classifier, loader)
        result = [args.name, round(test_accuracy, 4), round(test_f1, 4),
                  round(test_precision, 4), round(test_recall, 4)]
        update_leaderboard(result, args.leaderboard_path)
        logging.info(f"Test set {i} results: {result}")


def _build_name(args):
    from pathlib import Path
    name = '-'.join([''.join(Path(p).stem.split('-')[-1].split('_')) for p in args.path])
    name += f"_{'-'.join(args.norm)}_{'-'.join(args.model)}"
    name += f"_{args.pktcount}_{args.batch_size}_{args.epoch}_{args.lr}_{args.input_dim}_{args.hidden_size}_{args.num_layers}_{args.dropout}_{args.fusion_dim}_{args.fc_hidden_size}"
    return name

