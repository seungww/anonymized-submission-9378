import os
import csv
import argparse
import subprocess
import ipaddress
import logging
from datetime import datetime
from scapy.all import rdpcap, IP, TCP, UDP
import concurrent.futures

# ============================= #
#           Constants           #
# ============================= #
META_IP_LIST_PATH = 'misc/meta-ip-prefix.list'
DNS_IP_RANGE = ipaddress.ip_network('8.8.0.0/16')
DEFAULT_SRC_IPS = [
    '10.0.0.4', '10.0.0.5', '10.0.0.6', '10.0.1.1', '10.0.1.2', '10.0.1.3',
    '10.0.1.4', '10.0.1.5', '10.0.1.6', '10.0.1.7', '10.0.1.8', '10.0.0.11',
    '10.0.0.13', '10.0.0.15', '10.0.0.17', '10.0.0.19'
]
FEATURE_KEYS = [
    "PKT_index", "PKT_tag", "PKT_dir", "PKT_len", "PKT_ts", "PKT_dirlen",
    "IP_version", "IP_ihl", "IP_tos", "IP_len", "IP_id", "IP_flags", "IP_frag",
    "IP_ttl", "IP_proto", "IP_chksum", "IP_src", "IP_dst",
    "TCP_sport", "TCP_dport", "TCP_seq", "TCP_ack", "TCP_dataofs", "TCP_reserved",
    "TCP_flags", "TCP_window", "TCP_chksum", "TCP_urgptr",
    "UDP_sport", "UDP_dport", "UDP_len", "UDP_chksum"
]

# ============================= #
#         Logging Setup         #
# ============================= #
def init_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logging_extracting-metadata_{timestamp}.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# ============================= #
#       Utility Functions       #
# ============================= #
def get_all_files(target_dir):
    return sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(target_dir)
        for name in files if name.endswith('.pcap')
    )

def check_ip_in_prefix(ip, ip_prefix):
    ip_addr = ipaddress.ip_address(ip)
    return any(ip_addr in ip_range for ip_range in ip_prefix)

def get_meta_prefix_list(prefix_path):
    if not os.path.exists(prefix_path):
        logging.error(f"Meta IP list file not found: {prefix_path}")
        exit(1)
    with open(prefix_path) as f:
        return [ipaddress.ip_network(line.strip()) for line in f]

# ============================= #
#       Feature Extractor       #
# ============================= #
def extract_features(packets, opt):
    features = []
    meta_syn = True 
    dns_syn = others_syn = opt.all

    for idx, pkt in enumerate(packets, 1):
        row = [idx]
        if not pkt.haslayer(IP):
            continue

        ip_layer = pkt[IP]
        src, dst = ip_layer.src, ip_layer.dst
        direction = 1 if src in opt.ip_src_list else -1
        dirlen = pkt.len if direction == 1 else -pkt.len

        if check_ip_in_prefix(src, opt.meta_ip_prefix) or check_ip_in_prefix(dst, opt.meta_ip_prefix):
            if pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
                meta_syn = True
            if not meta_syn:
                continue
            tag = 'META'
        elif ipaddress.ip_address(src) in DNS_IP_RANGE or ipaddress.ip_address(dst) in DNS_IP_RANGE:
            if pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
                dns_syn = True
            if not dns_syn:
                continue
            tag = 'DNS'
        else:
            if pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
                others_syn = True
            if not others_syn:
                continue
            tag = 'Others'

        ip_info = [
            ip_layer.version, ip_layer.ihl, ip_layer.tos, ip_layer.len, ip_layer.id,
            str(ip_layer.flags), ip_layer.frag, ip_layer.ttl, ip_layer.proto,
            ip_layer.chksum, ip_layer.src, ip_layer.dst
        ]

        tcp_info = ([
            pkt[TCP].sport, pkt[TCP].dport, pkt[TCP].seq, pkt[TCP].ack,
            pkt[TCP].dataofs, pkt[TCP].reserved, str(pkt[TCP].flags),
            pkt[TCP].window, pkt[TCP].chksum, pkt[TCP].urgptr
        ] if pkt.haslayer(TCP) else ['NaN'] * 10)

        udp_info = ([
            pkt[UDP].sport, pkt[UDP].dport, pkt[UDP].len, pkt[UDP].chksum
        ] if pkt.haslayer(UDP) else ['NaN'] * 4)

        row.extend([tag, direction, pkt.len, pkt.time, dirlen])
        row.extend(ip_info + tcp_info + udp_info)
        features.append(row)

    return features

# ============================= #
#       Processing Logic        #
# ============================= #
def process_pcap_file(args):
    pcap_file, opt, feature_keys = args
    file_name = os.path.basename(pcap_file)
    label = os.path.basename(os.path.dirname(pcap_file))
    label_dir = os.path.join(opt.features_path, label)
    os.makedirs(label_dir, exist_ok=True)
    output_file = os.path.join(label_dir, os.path.splitext(file_name)[0] + '.feat')

    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        logging.error(f"Failed to read {pcap_file}: {e}")
        return None

    features = extract_features(packets, opt)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(feature_keys)
        writer.writerows(features)
    return output_file

def make_features(opt):
    files = get_all_files(opt.pcapdir)
    args_list = [(f, opt, FEATURE_KEYS) for f in files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for count, output_file in enumerate(executor.map(process_pcap_file, args_list), 1):
            if output_file:
                logging.info(f"[ {count} / {len(files)} ] Feature file created: {output_file}")

# ============================= #
#         Config & Main         #
# ============================= #
def initial_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcapdir', type=str, required=True, help='Dataset directory (pcap files)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--all', action='store_true', help='Include all packets (default: starting from first TCP SYN')
    opt = parser.parse_args()

    if not os.path.exists(opt.pcapdir):
        logging.error("Dataset directory does not exist.")
        exit(1)

    opt.output = opt.output.strip('/')
    opt.features_path = os.path.join("features", opt.output)
    opt.data_path = os.path.join("features", opt.output + ".csv")
    os.makedirs(opt.features_path, exist_ok=True)

    opt.ip_src_list = DEFAULT_SRC_IPS
    opt.meta_ip_prefix = get_meta_prefix_list(META_IP_LIST_PATH)
    return opt

def main():
    init_logging()
    opt = initial_config()
    make_features(opt)

if __name__ == '__main__':
    main()
