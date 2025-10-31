import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter
from sklearn.metrics import confusion_matrix


def evaluate(args, classifier, data_loader):
    classifier.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[:-1]
            batch_y = batch[-1]

            batch_x = [x.to(args.device) for x in batch_x]
            batch_y = batch_y.to(args.device)

            outputs = classifier(batch_x)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    logging.info(f"{args},{accuracy:.4f},{f1:.4f},{precision:.4f},{recall:.4f}")
    return accuracy, f1, precision, recall


def evaluate_with_attention(args, classifier, data_loader, names, max_samples_to_print=5):
    classifier.eval()
    all_preds, all_targets = [], []

    sample_global_index = 0 

    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[:-1]
            batch_y = batch[-1]

            batch_x = [x.to(args.device) for x in batch_x]
            batch_y = batch_y.to(args.device)

            # dual attention
            logits, attn_weights_list, feature_weights = classifier(
                batch_x, return_attn_weights=True, return_feature_weights=True
            )
            _, predicted = torch.max(logits, dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

            batch_size = batch_y.size(0)
            for i in range(batch_size):
                if sample_global_index >= max_samples_to_print:
                    break

                name = names[sample_global_index]
                true_label = batch_y[i].item()
                pred_label = predicted[i].item()

                logging.debug(f"[SAMPLE {sample_global_index}] name: {name}, True: {true_label}, Pred: {pred_label}")

                if args.feature_attention:
                    # Feature-wise attention
                    fw = feature_weights[i].cpu().numpy()  # [N]
                    top_f_idx = fw.argsort()[::-1]
                    top_f_scores = [f"Input {j} ({fw[j]:.4f})" for j in top_f_idx]
                    logging.debug("  [Feature Attention] Top Features:")
                    logging.debug("   " + ", ".join(top_f_scores))

                if args.timestep_attention:
                    # Timestep-wise attention per feature
                    for j, attn in enumerate(attn_weights_list):
                        weights = attn[i].cpu().numpy()
                        all_idx = weights.argsort()[::-1]
                        #top_idx = weights.argsort()[-10:][::-1]
                        #top_scores = [f"{idx} ({weights[idx]:.4f})" for idx in top_idx]
                        #prt_scores = ", ".join(top_scores)
                        
                        all_scores = [f"{idx} ({weights[idx]:.4f})" for idx in all_idx]
                        prt_scores = ", ".join(all_scores)
                        logging.debug(f"  [Input {j}] Top Timestep Attention:")
                        logging.debug(f"   {prt_scores}")

                sample_global_index += 1

    # confusion matrix
    # pred_dist = Counter(all_preds)
    # true_dist = Counter(all_targets)
    # logging.debug(f"Prediction distribution: {pred_dist}")
    # logging.debug(f"   True distribution: {true_dist}")

    # cm = confusion_matrix(all_targets, all_preds)
    # logging.debug(f"Confusion matrix:\n{cm}")

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    logging.info(f"{args},{accuracy:.4f},{f1:.4f},{precision:.4f},{recall:.4f}")
    return accuracy, f1, precision, recall


def evaluate_loss(classifier, data_loader, device, criterion):
    classifier.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[:-1]
            batch_y = batch[-1]

            batch_x = [x.to(device) for x in batch_x]
            batch_y = batch_y.to(device)

            outputs = classifier(batch_x)
            _, predicted = torch.max(outputs, 1)

            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    acc = 100 * correct / total
    return avg_loss, acc


def evaluate_accuracy(model, data_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[:-1]
            batch_y = batch[-1]

            batch_x = [x.to(device) for x in batch_x]
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    return accuracy_score(all_targets, all_preds)


def evaluate_openworld(args, classifier, data_loader, known_class_indices):
    classifier.eval()
    all_confs, all_preds, all_targets, all_known_flags = [], [], [], []

    with torch.no_grad():
        for batch in data_loader:
            batch_x = batch[:-1]
            batch_y = batch[-1]

            batch_x = [x.to(args.device) for x in batch_x]
            batch_y = batch_y.to(args.device)

            outputs = classifier(batch_x)
            
            temperature = 3
            probs = torch.softmax(outputs / temperature, dim=1)
            confs, predicted = torch.max(probs, dim=1)
            scores = confs

            #probs = torch.softmax(outputs, dim=1)
            #confs, predicted = torch.max(probs, dim=1)

            all_confs.extend(confs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_known_flags.extend([(1 if y.item() in known_class_indices else 0) for y in batch_y])

    all_confs = np.array(all_confs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_known_flags = np.array(all_known_flags)

    thresholds = list(np.arange(0, 1.0, 0.005)) + [0.999]
    results = []

    for thresh in thresholds:
        pred_labels = []
        for i in range(len(all_confs)):
            if all_confs[i] < thresh:
                pred_labels.append(-1)
            else:
                pred_labels.append(all_preds[i])
        pred_labels = np.array(pred_labels)

        known_mask = all_known_flags == 1
        cwa = accuracy_score(all_targets[known_mask], pred_labels[known_mask])

        unknown_mask = all_known_flags == 0
        true_unknown = (pred_labels[unknown_mask] == -1).sum()
        odr = true_unknown / unknown_mask.sum()

        false_unknown = ((pred_labels == -1) & (all_known_flags == 1)).sum()
        fpr = false_unknown / known_mask.sum()

        results.append((thresh, cwa, odr, fpr))
        logging.info(f"[Threshold: {thresh:.4f}] CWA: {cwa:.4f}, ODR: {odr:.4f}, FPR: {fpr:.4f}")

