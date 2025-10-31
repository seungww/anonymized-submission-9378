import logging
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold

from vrscanner.loader.data_loader import load_data
from vrscanner.loader.batch_loader import load_batch
from vrscanner.core.model_selection import get_model
from vrscanner.core.trainer import train
from vrscanner.core.evaluator import evaluate_openworld
from vrscanner.utils import update_leaderboard


def openworld_evaluation(args):
    logging.info(args)
    all_data, labels, class_names, timestamps, _ = load_data(args.path, args.pktcount)

    args.name = _build_name(args)
    norms = args.norm
    batch_size = args.batch_size
    kfold = args.kfold
    random_state = 42
    test_size = 0.2
    val_size = 0.125

    app_list = class_names.copy()
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    if kfold == 1:
        _evaluate_single_fold(
            args, all_data, labels, norms, batch_size, test_size, val_size,
            random_state, app_list, class_to_idx
        )
    else:
        _evaluate_cross_fold(
            args, all_data, labels, norms, batch_size, test_size, val_size,
            random_state, app_list, class_to_idx
        )


def _evaluate_single_fold(args, all_data, labels, norms, batch_size, test_size, val_size,
                          random_state, app_list, class_to_idx):
    train_data, val_data, test_data = [], [], []
    random.seed(random_state)
    random.shuffle(app_list)

    split_idx = int(len(app_list) * (1 - test_size))
    seen_apps = app_list[:split_idx]
    unseen_apps = app_list[split_idx:]

    seen_indices, unseen_indices = [], []
    for i, label in enumerate(labels):
        if label in [class_to_idx[app] for app in seen_apps]:
            seen_indices.append(i)
        else:
            unseen_indices.append(i)

    seen_indices = np.array(seen_indices)
    seen_train_idx, seen_test_idx = train_test_split(
        seen_indices, test_size=test_size, random_state=random_state, stratify=labels[seen_indices]
    )

    train_indices = seen_train_idx
    test_indices = np.concatenate([seen_test_idx, unseen_indices])

    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size, random_state=random_state, stratify=labels[train_indices]
    )

    for data in all_data:
        train_data.append(data[train_indices])
        val_data.append(data[val_indices])
        test_data.append(data[test_indices])

    y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]
    train_loader, val_loader, test_loader, scalers, token_dicts = load_batch(
        norms, (train_data, y_train), (val_data, y_val), (test_data, y_test), batch_size
    )

    num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) for i in range(len(norms))]
    classifier = get_model(args, len(class_to_idx), num_tokens_list)
    logging.info(f"Classifier: {classifier}")
    logging.info("Training...")
    classifier = train(args, classifier, train_loader, val_loader)

    known_class_indices = [class_to_idx[app] for app in seen_apps]
    evaluate_openworld(args, classifier, test_loader, known_class_indices)


def _evaluate_cross_fold(args, all_data, labels, norms, batch_size, test_size, val_size,
                         random_state, app_list, class_to_idx):
    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=random_state)

    for fold_idx, (seen_app_idx, unseen_app_idx) in enumerate(kf.split(app_list)):
        logging.info(f"=== Fold {fold_idx + 1} / {args.kfold} ===")
        seen_apps = [app_list[i] for i in seen_app_idx]
        unseen_apps = [app_list[i] for i in unseen_app_idx]

        seen_indices, unseen_indices = [], []
        for i, label in enumerate(labels):
            if label in [class_to_idx[app] for app in seen_apps]:
                seen_indices.append(i)
            else:
                unseen_indices.append(i)

        seen_indices = np.array(seen_indices)
        seen_train_idx, seen_test_idx = train_test_split(
            seen_indices, test_size=test_size, random_state=random_state + fold_idx, stratify=labels[seen_indices]
        )

        train_indices = seen_train_idx
        test_indices = np.concatenate([seen_test_idx, unseen_indices])

        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size, random_state=random_state + fold_idx, stratify=labels[train_indices]
        )

        train_data, val_data, test_data = [], [], []
        for data in all_data:
            train_data.append(data[train_indices])
            val_data.append(data[val_indices])
            test_data.append(data[test_indices])

        y_train, y_val, y_test = labels[train_indices], labels[val_indices], labels[test_indices]
        train_loader, val_loader, test_loader, scalers, token_dicts = load_batch(
            norms, (train_data, y_train), (val_data, y_val), (test_data, y_test), batch_size
        )

        num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) for i in range(len(norms))]
        classifier = get_model(args, len(class_to_idx), num_tokens_list)
        logging.info(f"Classifier: {classifier}")
        logging.info("Training...")
        classifier = train(args, classifier, train_loader, val_loader)

        known_class_indices = [class_to_idx[app] for app in seen_apps]
        evaluate_openworld(args, classifier, test_loader, known_class_indices)


def _build_name(args):
    name = '-'.join([''.join(Path(p).stem.split('-')[-1].split('_')) for p in args.path])
    name += f"_{'-'.join(args.norm)}_{'-'.join(args.model)}"
    name += f"_{args.pktcount}_{args.batch_size}_{args.epoch}_{args.lr}_{args.input_dim}_{args.hidden_size}_{args.num_layers}_{args.dropout}_{args.fusion_dim}_{args.fc_hidden_size}"
    return name

