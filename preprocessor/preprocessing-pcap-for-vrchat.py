import os
import argparse
import ipaddress
import logging
from datetime import datetime
from scapy.all import rdpcap, wrpcap, IP, TCP, UDP
from scapy.error import Scapy_Exception
import concurrent.futures

# ============================= #
#           Constants           #
# ============================= #
COMMAND_IP = ipaddress.ip_address('10.0.0.1')
DNS_IP_RANGE = ipaddress.ip_network('8.8.0.0/16')
DHCP_IP_RANGE = ipaddress.ip_network('255.255.0.0/16')
META_IP_LIST_PATH = 'misc/meta-ip-prefix.list'
LOCAL_NETWORKS = [
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv4Network("240.0.0.0/4"),
]

# ============================= #
#         Logging Setup         #
# ============================= #
def init_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logging_preprocessing-pcap_{timestamp}.log"
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
def check_ip_in_prefix(ip, ip_prefix):
    ip_addr = ipaddress.ip_address(ip)
    return any(ip_addr in ip_range for ip_range in ip_prefix)

def check_local_networks(ip):
    ip_addr = ipaddress.ip_address(ip)
    return any(ip_addr in net for net in LOCAL_NETWORKS)

def get_all_files(target_dir):
    return sorted(
        os.path.join(root, name)
        for root, _, files in os.walk(target_dir)
        for name in files
    )

def get_ip_pair(pkt):
    if not pkt.haslayer(IP):
        return None, None
    return pkt[IP].src, pkt[IP].dst

# ============================= #
#       Packet Extractors       #
# ============================= #
# Only For VRChat Traffic
def extract_full_packets(packets):
    logging.info("Extracting full packets")
    start_flag = False
    udp_start_flag = False
    extracted = []
    for pkt in packets:
        ip_src, ip_dst = get_ip_pair(pkt)
        if not ip_src:
            continue

        if COMMAND_IP.exploded in [ip_src, ip_dst]:
            if pkt.haslayer(TCP) and pkt[TCP].payload:
                payload = pkt[TCP].payload.load.decode('utf-8', errors='ignore')
                if 'am start' in payload:
                    start_flag = True
                elif 'am force-stop' in payload:
                    start_flag = False
                    break

        if pkt.haslayer(UDP):
            udp_layer = pkt[UDP]
            if udp_layer.sport == 5055 or udp_layer.dport == 5055:
                udp_start_flag = True

        if start_flag and udp_start_flag:
            ip_src_addr = ipaddress.ip_address(ip_src)
            ip_dst_addr = ipaddress.ip_address(ip_dst)
            if ip_src_addr == COMMAND_IP or ip_dst_addr == COMMAND_IP:
                continue
            if ip_src_addr in DHCP_IP_RANGE or ip_dst_addr in DHCP_IP_RANGE:
                continue
            extracted.append(pkt)
    return extracted

def extract_app_and_meta_packets(packets, include_all_before_syn, meta_ip_prefix):
    logging.info("Extracting app-specific and platform-wise packets")
    start_flag = include_all_before_syn
    extracted = []

    for pkt in packets:
        ip_src, ip_dst = get_ip_pair(pkt)
        if not ip_src:
            continue

        if ipaddress.ip_address(ip_src) in DNS_IP_RANGE or ipaddress.ip_address(ip_dst) in DNS_IP_RANGE:
            continue
        if check_local_networks(ip_src) or check_local_networks(ip_dst):
            continue

        if not start_flag and pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
            start_flag = True

        if start_flag:
            extracted.append(pkt)
    return extracted

def extract_app_packets(packets, include_all_before_syn, meta_ip_prefix):
    logging.info("Extracting app-specific packets")
    start_flag = include_all_before_syn
    extracted = []

    for pkt in packets:
        ip_src, ip_dst = get_ip_pair(pkt)
        if not ip_src:
            continue

        if check_ip_in_prefix(ip_src, meta_ip_prefix) or check_ip_in_prefix(ip_dst, meta_ip_prefix):
            continue
        if ipaddress.ip_address(ip_src) in DNS_IP_RANGE or ipaddress.ip_address(ip_dst) in DNS_IP_RANGE:
            continue
        if check_local_networks(ip_src) or check_local_networks(ip_dst):
            continue

        if not start_flag and pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
            start_flag = True

        if start_flag:
            extracted.append(pkt)
    return extracted

def extract_meta_packets(packets, include_all_before_syn, meta_ip_prefix):
    logging.info("Extracting platform packets")
    start_flag = include_all_before_syn
    extracted = []

    for pkt in packets:
        ip_src, ip_dst = get_ip_pair(pkt)
        if not ip_src:
            continue

        if check_ip_in_prefix(ip_src, meta_ip_prefix) or check_ip_in_prefix(ip_dst, meta_ip_prefix):
            if check_local_networks(ip_src) or check_local_networks(ip_dst):
                continue
            if not start_flag and pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
                start_flag = True
            if start_flag:
                extracted.append(pkt)
    return extracted

def extract_dns_packets(packets, include_all_before_syn):
    logging.info("Extracting DNS packets")
    start_flag = include_all_before_syn
    extracted = []

    for pkt in packets:
        ip_src, ip_dst = get_ip_pair(pkt)
        if not ip_src:
            continue

        if ipaddress.ip_address(ip_src) in DNS_IP_RANGE or ipaddress.ip_address(ip_dst) in DNS_IP_RANGE:
            if check_local_networks(ip_src) or check_local_networks(ip_dst):
                continue
            if not start_flag and pkt.haslayer(TCP) and pkt[TCP].flags == 'S':
                start_flag = True
            if start_flag:
                extracted.append(pkt)
    return extracted

# ============================= #
#        PCAP Processing        #
# ============================= #
def process_single_pcap(args):
    pcap_file, opt = args
    file_name = os.path.basename(pcap_file)
    label = os.path.basename(os.path.dirname(pcap_file))
    label_dir = os.path.join(opt.output, label)
    os.makedirs(label_dir, exist_ok=True)

    try:
        packets = rdpcap(pcap_file)
    except Scapy_Exception as e:
        return f"[ERROR] Failed to read {pcap_file}: {e}"

    if opt.etype == 'full':
        app_packets = extract_full_packets(packets)
    elif opt.etype == 'app-and-meta':
        app_packets = extract_app_and_meta_packets(packets, opt.all, opt.meta_ip_prefix)
    elif opt.etype == 'app':
        app_packets = extract_app_packets(packets, opt.all, opt.meta_ip_prefix)
    elif opt.etype == 'meta':
        app_packets = extract_meta_packets(packets, opt.all, opt.meta_ip_prefix)
    elif opt.etype == 'dns':
        app_packets = extract_dns_packets(packets, opt.all)
    else:
        raise ValueError(f"Invalid etype encountered during processing: {opt.etype}")

    if app_packets:
        output_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + '.pcap')
        wrpcap(output_path, app_packets)
        return f"Saved: {output_path}"
    else:
        return f"[WARN] No packets extracted: {pcap_file}"

def safe_process(args):
    try:
        return process_single_pcap(args)
    except Exception as e:
        return f"[ERROR] Exception in {args[0]}: {e}"

def make_new_pcap(opt):
    files = get_all_files(opt.pcapdir)
    args_list = [(pcap_file, opt) for pcap_file in files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for count, result in enumerate(executor.map(safe_process, args_list), 1):
            logging.info(f"[ {count} / {len(files)} ] {result}")

# ============================= #
#         Config & Main         #
# ============================= #
def initial_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcapdir', type=str, required=True, help='Dataset directory (pcap files)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--etype', type=str, required=True, help='Extract type (full, app-and-meta, app, meta, dns)')
    parser.add_argument('--all', action='store_true', help='Include all packets (default: starting from first TCP SYN')
    opt = parser.parse_args()

    if not os.path.exists(opt.pcapdir):
        logging.error("Dataset directory does not exist.")
        exit()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    if opt.etype not in ['full', 'app-and-meta', 'app', 'meta', 'dns']:
        logging.error("Invalid etype. Choose from: full, app-and-meta, app, meta, dns")
        exit()

    opt.meta_ip_prefix = []
    if opt.etype in ['app', 'meta', 'app-and-meta']:
        if not os.path.exists(META_IP_LIST_PATH):
            logging.error(f"Meta IP list file not found: {META_IP_LIST_PATH}")
        else:
            with open(META_IP_LIST_PATH) as f:
                opt.meta_ip_prefix = [ipaddress.ip_network(line.strip()) for line in f]

    return opt

def main():
    init_logging()
    opt = initial_config()
    make_new_pcap(opt)

if __name__ == '__main__':
    main()

