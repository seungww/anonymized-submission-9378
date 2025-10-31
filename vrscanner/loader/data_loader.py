import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def load_data(paths, pktcount):
    all_data = []
    labels = None
    class_names = None
    all_names = []

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
        all_names.extend(df["name"].tolist())
        df["label"] = df["name"].astype(str).apply(lambda x: x.split("_")[0] if "_" in x else "unknown")
        df["timestamp"] = df["name"].str.split("_").str[-1].astype(int)

        if labels is None:
            labels = df["label"].values
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
            class_names = label_encoder.classes_
            timestamps = df["timestamp"].values

        data = df.drop(columns=["name", "label", "timestamp"]).values
        all_data.append(data)

        logging.info(f"Processed data from {path}:")
        logging.info(data.shape)
        logging.info(data)

    # df_all = pd.DataFrame({'name': all_names})
    # df_all['label'] = df_all['name'].str.split('_').str[0]
    # counts = df_all['label'].value_counts().sort_index()

    # print("=== Label Count ===")
    # print(counts.to_string())
    # print(f"Total: {len(df_all)}")

    return all_data, labels, class_names, timestamps, all_names

