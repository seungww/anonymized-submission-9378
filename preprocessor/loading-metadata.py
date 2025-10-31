import os
import csv
import argparse
import ipaddress
import logging
from datetime import datetime
import pandas as pd

# ============================= #
#           Constants           #
# ============================= #
LOCAL_NETWORKS = [
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv4Network("240.0.0.0/4"),
]

PTYPE_MAP = {
    'app-and-meta': ['META', 'Others'],
    'meta': ['META'],
    'dns': ['DNS'],
    'app': ['Others']
}

# ============================= #
#         Logging Setup         #
# ============================= #
def init_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logging_loading-metadata_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# ============================= #
#       Utility Functions       #
# ============================= #
def get_all_files(target):
    return sorted(
        os.path.join(root, file_name)
        for root, _, file_names in os.walk(target)
        for file_name in file_names if file_name.endswith('.feat')
    )

def is_local(ip_str):
    try:
        ip = ipaddress.IPv4Address(ip_str)
        return any(ip in net for net in LOCAL_NETWORKS)
    except ValueError:
        return False

# ============================= #
#       Processing Logic        #
# ============================= #
def loading(opt):
    files = get_all_files(opt.featdir)
    with open(opt.data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['name'] + list(range(1, opt.pktcount + 1))
        writer.writerow(header)

        for count, feat_file in enumerate(files, start=1):
            file_name = os.path.splitext(os.path.basename(feat_file))[0]
            df = pd.read_csv(feat_file, dtype=str)

            mask = ~df['IP_src'].apply(is_local) & ~df['IP_dst'].apply(is_local)
            df = df[mask]

            if opt.ptype != 'full':
                df = df[df['PKT_tag'].isin(PTYPE_MAP[opt.ptype])]

            if opt.dtype != 'both':
                df = df[df['PKT_dir'] == ('1' if opt.dtype == 'out' else '-1')]

            if len(df) <= 1:
                logging.warning(f"No usable features in: {feat_file}")
                continue

            if 'PKT_ts' in df.columns:
                df['PKT_ts'] = df['PKT_ts'].astype(float).round(6).apply(lambda x: f"{x:.6f}")

            if 'PKT_ts_diff' in opt.feature_keys and 'PKT_ts' in df.columns:
                df['PKT_ts_diff'] = df['PKT_ts'].astype(float).diff().fillna(0).round(6).apply(lambda x: f"{x:.6f}")

            if 'PKT_ts_rel' in opt.feature_keys and 'PKT_ts' in df.columns:
                first_pkt_ts = df['PKT_ts'].astype(float).iloc[0]
                df['PKT_ts_rel'] = df['PKT_ts'].astype(float).sub(first_pkt_ts).round(6).apply(lambda x: f"{x:.6f}")

            row = [file_name]
            values = df[opt.feature_keys].fillna("").astype(str).agg('_'.join, axis=1)
            row.extend(values[:opt.pktcount])
            logging.info(f"[ {count} / {len(files)} ] Inserted vector: {file_name}")
            writer.writerow(row)

# ============================= #
#         Config & Main         #
# ============================= #
def initial_config():
    feature_keys = [
        "PKT_index", "PKT_tag", "PKT_dir", "PKT_len", "PKT_ts", "PKT_ts_diff", "PKT_ts_rel", "PKT_dirlen",
        "IP_version", "IP_ihl", "IP_tos", "IP_len", "IP_id", "IP_flags", "IP_frag",
        "IP_ttl", "IP_proto", "IP_chksum", "IP_src", "IP_dst",
        "TCP_sport", "TCP_dport", "TCP_seq", "TCP_ack", "TCP_dataofs", "TCP_reserved",
        "TCP_flags", "TCP_window", "TCP_chksum", "TCP_urgptr",
        "UDP_sport", "UDP_dport", "UDP_len", "UDP_chksum"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--featdir', type=str, required=True, help='Dataset directory (feat files)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--ptype', type=str.lower, required=True, choices=['full', 'app-and-meta', 'app', 'meta', 'dns'], help='Packet type to include')
    parser.add_argument('--dtype', type=str.lower, default='both', choices=['in', 'out', 'both'], help='Packet direction')
    parser.add_argument('--pktcount', type=int, default=10000, help='Number of packets to include')

    for key in feature_keys:
        parser.add_argument(f'--{key}', action='store_true', help=key)

    opt = parser.parse_args()
    opt.feature_keys = [k for k in feature_keys if getattr(opt, k)]

    if not os.path.exists(opt.featdir):
        logging.error("Feature directory does not exist.")
        exit(1)

    opt.output= opt.output.strip('/')
    opt.output_path = os.path.join("output", opt.output)
    opt.data_path = os.path.join(opt.output_path, opt.output + '.csv')
    os.makedirs(opt.output_path, exist_ok=True)

    return opt

def main():
    init_logging()
    opt = initial_config()
    loading(opt)

if __name__ == '__main__':
    main()

