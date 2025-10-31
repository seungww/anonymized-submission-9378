import logging
import numpy as np
from pathlib import Path
from vrscanner.loader.data_loader import load_data
from vrscanner.core.trainer import train_kfold


def process_step1(args):
    args.metadata = Path(args.path[0]).stem.split('_')[-1]

    logging.info(args)
    all_data, labels, class_names, _, all_names = load_data(args.path, args.pktcount)

    norm_list = ["none", "minmax", "zscore", "token", "binary", "maxabs", "l1norm", "l2norm", "power", "quantile", "robust"]
    if args.metadata in ['src', 'dst', 'flags']:
        norm_list = ["token"]

    norm_results = []
    for norm in norm_list:
        args.norm = [norm]
        args.name = _build_name(args)
        results = train_kfold(args, all_data, labels, class_names, all_names)
        norm_results.append(results)

    best_idx, best_f1 = -1, -1
    for i, result in enumerate(norm_results):
        acc, f1, pre, rec = result[0]
        if best_f1 < f1:
            best_idx, best_f1, best_result = i, f1, (acc, f1, pre, rec)

    logging.info(f"best({args.metadata}): {norm_list[best_idx]}, "
                 f"accuracy: {best_result[0]:.4f}, f1: {best_result[1]:.4f}, "
                 f"precision: {best_result[2]:.4f}, recall: {best_result[3]:.4f}")


def process_step2(args):
    args.metadata = Path(args.path[0]).stem.split('_')[-1]

    logging.info(args)
    all_data, labels, class_names, _, all_names = load_data(args.path, args.pktcount)

    model_list = ["gru", "bigru", "lstm", "bilstm", "rnn", "birnn"]
    model_results = []
    for model in model_list:
        args.model = [model]
        args.name = _build_name(args)
        results = train_kfold(args, all_data, labels, class_names, all_names)
        model_results.append(results)

    best_idx, best_f1 = -1, -1
    for i, result in enumerate(model_results):
        acc, f1, pre, rec = result[0]
        if best_f1 < f1:
            best_idx, best_f1, best_result = i, f1, (acc, f1, pre, rec)

    logging.info(f"best({args.metadata}): {model_list[best_idx]}, "
                 f"accuracy: {best_result[0]:.4f}, f1: {best_result[1]:.4f}, "
                 f"precision: {best_result[2]:.4f}, recall: {best_result[3]:.4f}")


def process_step3(args):
    logging.info(args)
    all_data, labels, class_names, _, all_names = load_data(args.path, args.pktcount)

    args.name = _build_name(args)
    results = train_kfold(args, all_data, labels, class_names, all_names)
    logging.info(results)


def process_train(args):
    logging.info(args)
    all_data, labels, class_names, _, all_names = load_data(args.path, args.pktcount)

    args.name = _build_name(args)
    results = train_kfold(args, all_data, labels, class_names, all_names)
    logging.info(results)


def sliding_window_evaluation(args):
    logging.info(args)
    all_data, labels, class_names, _, all_names = load_data(args.path, args.pktcount)

    for start in range(0, args.pktcount - args.window_size + 1, args.step_size):
        end = start + args.window_size
        logging.info(f"Training from {start} to {end} / {args.pktcount}")
        args.name = f"{_build_name(args)}_{args.window_size}_{args.step_size}_{start}"
        sliced = []
        for array in all_data:
            sliced_array = array[:, start:end].tolist()
            sliced.append(sliced_array)
        results = train_kfold(args, sliced, labels, class_names, all_names)
        logging.info(results)


def _build_name(args):
    name = '-'.join([''.join(Path(p).stem.split('-')[-1].split('_')) for p in args.path])
    name += f"_{'-'.join(args.norm)}_{'-'.join(args.model)}"
    name += f"_{args.pktcount}_{args.batch_size}_{args.epoch}_{args.lr}_{args.input_dim}_{args.hidden_size}_{args.num_layers}_{args.dropout}_{args.fusion_dim}_{args.fc_hidden_size}"
    return name

