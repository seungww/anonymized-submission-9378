import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from vrscanner.loader.data_loader import load_data
from vrscanner.loader.batch_loader import load_batch
from vrscanner.core.model_selection import get_model
from vrscanner.core.trainer import train
from vrscanner.core.evaluator import evaluate_openworld


def app_launch_detection_evaluation(args):
    logging.info(args)
    all_data, labels, class_names, timestamps, _ = load_data(args.path, args.pktcount)

    if not args.unknown_path:
        raise ValueError("unknown_path is required for app launch detection evaluation.")

    unknown_data = _load_unlabeled_data(args.unknown_path, args.pktcount)
    if len(unknown_data) != len(all_data):
        raise ValueError("unknown_path count must match the number of --path inputs.")

    args.name = _build_name(args)
    norms = args.norm
    batch_size = args.batch_size
    random_state = 42
    test_size = 0.2
    val_size = 0.125

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=random_state, stratify=labels[train_idx]
    )

    train_data, val_data, test_data = [], [], []
    for data in all_data:
        train_data.append(data[train_idx])
        val_data.append(data[val_idx])
        test_data.append(data[test_idx])

    y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

    unknown_count = unknown_data[0].shape[0] if unknown_data else 0
    if unknown_count > 0:
        for i in range(len(test_data)):
            test_data[i] = np.concatenate([test_data[i], unknown_data[i]], axis=0)
        unknown_labels = np.full(unknown_count, -1, dtype=y_test.dtype)
        y_test = np.concatenate([y_test, unknown_labels], axis=0)

    train_loader, val_loader, test_loader, scalers, token_dicts = load_batch(
        norms, (train_data, y_train), (val_data, y_val), (test_data, y_test), batch_size
    )

    num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) for i in range(len(norms))]
    classifier = get_model(args, len(class_names), num_tokens_list)
    logging.info(f"Classifier: {classifier}")
    logging.info("Training...")
    classifier = train(args, classifier, train_loader, val_loader)

    known_class_indices = list(range(len(class_names)))
    evaluate_openworld(args, classifier, test_loader, known_class_indices)


def _load_unlabeled_data(paths, pktcount):
    all_data = []
    for path in paths:
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception as e:
            logging.warning(f"Error reading the entire file: {e}")
            logging.warning("Reading the file in chunks...")
            chunks = pd.read_csv(path, dtype=str, chunksize=50)
            df = pd.concat(chunks, ignore_index=True)

        df = df.convert_dtypes()
        max_columns = df.shape[1] - 1
        if pktcount > max_columns:
            raise ValueError(f"pktcount ({pktcount}) exceeds available packet columns ({max_columns}).")

        data_columns = [str(i) for i in range(1, pktcount + 1)]
        selected_columns = ["name"] + data_columns
        df = df[selected_columns]

        data = df.drop(columns=["name"]).values
        all_data.append(data)

        logging.info(f"Processed unknown data from {path}:")
        logging.info(data.shape)
        logging.info(data)

    return all_data


def _build_name(args):
    name = '-'.join([''.join(Path(p).stem.split('-')[-1].split('_')) for p in args.path])
    name += f"_{'-'.join(args.norm)}_{'-'.join(args.model)}"
    name += f"_{args.pktcount}_{args.batch_size}_{args.epoch}_{args.lr}_{args.input_dim}_{args.hidden_size}_{args.num_layers}_{args.dropout}_{args.fusion_dim}_{args.fc_hidden_size}"
    return name
