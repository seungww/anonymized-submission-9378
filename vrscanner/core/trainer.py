import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from collections import Counter

from vrscanner.core.evaluator import evaluate_loss, evaluate, evaluate_with_attention
from vrscanner.core.model_selection import get_model
from vrscanner.loader.batch_loader import load_batch
from vrscanner.utils import update_leaderboard


def train(args, classifier, train_loader, val_loader):
    learning_rate = args.lr
    epochs = args.epoch

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    classifier.to(args.device)

    best_val_loss = float('inf')
    best_model_weights = None

    try:
        for epoch in range(epochs):
            classifier.train()
            total_loss = 0

            for batch in train_loader:
                batch_x = batch[:-1]
                batch_y = batch[-1]

                batch_x = [x.to(args.device) for x in batch_x]
                batch_y = batch_y.to(args.device)

                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            val_loss, val_acc = evaluate_loss(classifier, val_loader, args.device, criterion)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = classifier.state_dict().copy()

            logging.info(
                f"Epoch {epoch + 1}/{epochs}, ValAcc: {val_acc:.2f}%, "
                f"TrainLoss: {train_loss:.4f}, ValLoss: {val_loss:.4f}, LR: {current_lr}"
            )

            if epoch >= 5 and val_acc < 1:
                logging.info(f"ValAcc too low {val_acc:.2f}%. Stopping early.")
                break

            if current_lr < 1e-4:
                logging.info(f"Learning rate {current_lr:.6f} is below threshold. Stopping early.")
                break

    except KeyboardInterrupt:
        logging.info("\nStopped")

    if best_model_weights is not None:
        classifier.load_state_dict(best_model_weights)

    return classifier


def train_kfold(args, all_data, labels, class_names, all_names, test_size=0.2, val_size=0.125, random_state=42):
    kfold = args.kfold
    norms = args.norm
    batch_size = args.batch_size
    performance_results = []

    if kfold == 1:
        train_data, val_data, test_data = [], [], []

        x_train_val_idx, x_test_idx = train_test_split(
            np.arange(len(all_data[0])), test_size=test_size, random_state=random_state
        )
        x_train_idx, x_val_idx = train_test_split(
            x_train_val_idx, test_size=val_size, random_state=random_state
        )

        train_names = [all_names[i] for i in x_train_idx]
        val_names = [all_names[i] for i in x_val_idx]
        test_names = [all_names[i] for i in x_test_idx]

        for data in all_data:
            x_train_val, x_test, y_train_val, y_test = train_test_split(
                data, labels, test_size=test_size, random_state=random_state
            )
            x_train, x_val, y_train, y_val = train_test_split(
                x_train_val, y_train_val, test_size=val_size, random_state=random_state
            )

            train_data.append(x_train)
            val_data.append(x_val)
            test_data.append(x_test)

        train_loader, val_loader, test_loader, scalers, token_dicts = load_batch(
            norms, (train_data, y_train), (val_data, y_val), (test_data, y_test), batch_size
        )
        num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) for i in range(len(norms))]

        classifier = get_model(args, len(class_names), num_tokens_list)
        logging.info(f"Classifier: {classifier}")
        logging.info("Training...")
        classifier = train(args, classifier, train_loader, val_loader)

        if args.strict:
            max_retries = 5  # bad local minimum
            for attempt in range(max_retries):
                val_loss, val_acc = evaluate_loss(classifier, val_loader, args.device, nn.CrossEntropyLoss())
                if val_acc > 2:
                    break
                elif attempt < max_retries:
                    logging.warning(f"ValAcc too low ({val_acc:.2f}%), getting stuck in a bad local minimum...")
                else:
                    logging.error("Training failed after maximum retries.")

                logging.info(f"Retraining with new initialization: attempt {attempt + 1}...")
                classifier = get_model(args, len(class_names), num_tokens_list)
                classifier = train(args, classifier, train_loader, val_loader)

        if args.feature_attention or args.timestep_attention:
            test_accuracy, test_f1, test_precision, test_recall = evaluate_with_attention(args, classifier, test_loader, names=test_names, max_samples_to_print=100000)
        else:
            test_accuracy, test_f1, test_precision, test_recall = evaluate(args, classifier, test_loader)

        performance_results.append((test_accuracy, test_f1, test_precision, test_recall))

        result = [args.name, round(test_accuracy, 4), round(test_f1, 4),
                  round(test_precision, 4), round(test_recall, 4)]
        update_leaderboard(result, args.leaderboard_path)

    else:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_data[0], labels)):
            logging.info(f"Training Fold {fold + 1}/{kfold}")

            train_val_data = [data[train_val_idx] for data in all_data]
            test_data = [data[test_idx] for data in all_data]
            y_train_val, y_test = labels[train_val_idx], labels[test_idx]

            test_names = [all_names[i] for i in test_idx]

            train_idx, val_idx = train_test_split(
                np.arange(len(y_train_val)), test_size=val_size, random_state=random_state
            )

            train_data = [data[train_idx] for data in train_val_data]
            val_data = [data[val_idx] for data in train_val_data]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            train_loader, val_loader, test_loader, scalers, token_dicts = load_batch(
                norms, (train_data, y_train), (val_data, y_val), (test_data, y_test), batch_size
            )
            num_tokens_list = [None if i not in token_dicts else len(token_dicts[i]) + 1 for i in range(len(norms))]

            classifier = get_model(args, len(class_names), num_tokens_list)
            logging.info(f"Classifier: {classifier}")
            logging.info("Training...")
            classifier = train(args, classifier, train_loader, val_loader)

            if args.strict:
                max_retries = 5  # bad local minimum
                for attempt in range(max_retries):
                    val_loss, val_acc = evaluate_loss(classifier, val_loader, args.device, nn.CrossEntropyLoss())
                    if val_acc > 2:
                        break
                    elif attempt < max_retries:
                        logging.warning(f"ValAcc too low ({val_acc:.2f}%), getting stuck in a bad local minimum...")
                    else:
                        logging.error("Training failed after maximum retries.")

                    logging.info(f"Retraining with new initialization: attempt {attempt + 1}...")
                    classifier = get_model(args, len(class_names), num_tokens_list)
                    classifier = train(args, classifier, train_loader, val_loader)

            if args.feature_attention or args.timestep_attention:
                test_accuracy, test_f1, test_precision, test_recall = evaluate_with_attention(args, classifier, test_loader, names=test_names, max_samples_to_print=100000)
            else:
                test_accuracy, test_f1, test_precision, test_recall = evaluate(args, classifier, test_loader)

            performance_results.append((test_accuracy, test_f1, test_precision, test_recall))

            result = [args.name, round(test_accuracy, 4), round(test_f1, 4),
                      round(test_precision, 4), round(test_recall, 4)]
            update_leaderboard(result, args.leaderboard_path)

    return performance_results

