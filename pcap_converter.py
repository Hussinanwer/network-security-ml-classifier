"""
PCAP to CSV converter for network traffic classification.
Extracts all 35 features from PCAP files for prediction.
Uses scapy instead of pyshark to avoid asyncio issues in Streamlit.
"""

from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from collections import defaultdict
import tempfile
import os


def pcap_to_dataframe(pcap_file_path):
    """
    Convert a PCAP file to a DataFrame with all 35 features.
    For use in prediction (no label needed).
    Uses scapy for synchronous packet processing (no asyncio issues).

    Args:
        pcap_file_path (str): Path to the PCAP file

    Returns:
        pd.DataFrame: DataFrame with extracted features (no label column)
    """
    if not os.path.exists(pcap_file_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file_path}")

    print(f"[*] Reading PCAP file with scapy...")

    # Read PCAP file using scapy (synchronous, no event loop needed)
    try:
        packets = rdpcap(pcap_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read PCAP file: {e}")

    print(f"[*] Loaded {len(packets)} packets")

    # Track flows
    flows = defaultdict(lambda: {
        'packets': 0,
        'bytes': 0,
        'packet_sizes': [],
        'packet_times': [],
        'syn_count': 0,
        'ack_count': 0,
        'fin_count': 0,
        'rst_count': 0,
        'psh_count': 0,
        'urg_count': 0,
        'start_time': None,
        'end_time': None,
        'src_ip': None,
        'dst_ip': None,
        'src_port': None,
        'dst_port': None,
        'protocol': None,
        'forward_packets': 0,
        'backward_packets': 0,
        'forward_bytes': 0,
        'backward_bytes': 0,
    })

    packet_count = 0

    # Process packets
    for packet in packets:
        packet_count += 1

        try:
            # Only process IP packets
            if not packet.haslayer(IP):
                continue

            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            packet_length = len(packet)
            timestamp = float(packet.time)

            # Process TCP packets
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                protocol = 'TCP'
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport

                # Create flow key (bidirectional)
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    direction = 'forward'
                    flow_src_ip, flow_dst_ip = src_ip, dst_ip
                    flow_src_port, flow_dst_port = src_port, dst_port
                else:
                    flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)
                    direction = 'backward'
                    flow_src_ip, flow_dst_ip = dst_ip, src_ip
                    flow_src_port, flow_dst_port = dst_port, src_port

                flow = flows[flow_key]
                flow['packets'] += 1
                flow['bytes'] += packet_length
                flow['packet_sizes'].append(packet_length)

                if direction == 'forward':
                    flow['forward_packets'] += 1
                    flow['forward_bytes'] += packet_length
                else:
                    flow['backward_packets'] += 1
                    flow['backward_bytes'] += packet_length

                flow['src_ip'] = flow_src_ip
                flow['dst_ip'] = flow_dst_ip
                flow['src_port'] = flow_src_port
                flow['dst_port'] = flow_dst_port
                flow['protocol'] = protocol

                flow['packet_times'].append(timestamp)
                if flow['start_time'] is None:
                    flow['start_time'] = timestamp
                flow['end_time'] = timestamp

                # TCP flags
                flags = tcp_layer.flags
                if flags.S: flow['syn_count'] += 1  # SYN
                if flags.A: flow['ack_count'] += 1  # ACK
                if flags.F: flow['fin_count'] += 1  # FIN
                if flags.R: flow['rst_count'] += 1  # RST
                if flags.P: flow['psh_count'] += 1  # PSH
                if flags.U: flow['urg_count'] += 1  # URG

            # Process UDP packets
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                protocol = 'UDP'
                src_port = udp_layer.sport
                dst_port = udp_layer.dport

                # Create flow key (bidirectional)
                if (src_ip, src_port) < (dst_ip, dst_port):
                    flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    direction = 'forward'
                    flow_src_ip, flow_dst_ip = src_ip, dst_ip
                    flow_src_port, flow_dst_port = src_port, dst_port
                else:
                    flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)
                    direction = 'backward'
                    flow_src_ip, flow_dst_ip = dst_ip, src_ip
                    flow_src_port, flow_dst_port = dst_port, src_port

                flow = flows[flow_key]
                flow['packets'] += 1
                flow['bytes'] += packet_length
                flow['packet_sizes'].append(packet_length)

                if direction == 'forward':
                    flow['forward_packets'] += 1
                    flow['forward_bytes'] += packet_length
                else:
                    flow['backward_packets'] += 1
                    flow['backward_bytes'] += packet_length

                flow['src_ip'] = flow_src_ip
                flow['dst_ip'] = flow_dst_ip
                flow['src_port'] = flow_src_port
                flow['dst_port'] = flow_dst_port
                flow['protocol'] = protocol

                flow['packet_times'].append(timestamp)
                if flow['start_time'] is None:
                    flow['start_time'] = timestamp
                flow['end_time'] = timestamp

        except (AttributeError, ValueError, KeyError):
            continue

    print(f"[+] Processed {packet_count} packets, found {len(flows)} flows")

    # Convert flows to feature list
    features_list = []

    for flow_key, flow_data in flows.items():
        if flow_data['packets'] == 0:
            continue

        # Calculate duration
        duration = 0
        if flow_data['start_time'] and flow_data['end_time']:
            duration = flow_data['end_time'] - flow_data['start_time']

        packet_sizes = flow_data['packet_sizes']

        # Calculate Inter-Arrival Times (IAT)
        packet_times = sorted(flow_data['packet_times'])
        iats = []
        if len(packet_times) > 1:
            for i in range(1, len(packet_times)):
                iats.append(packet_times[i] - packet_times[i-1])

        # Calculate rates and ratios
        packets_per_second = flow_data['packets'] / duration if duration > 0 else 0
        bytes_per_second = flow_data['bytes'] / duration if duration > 0 else 0
        bytes_per_packet = flow_data['bytes'] / flow_data['packets'] if flow_data['packets'] > 0 else 0

        syn_ack_ratio = flow_data['syn_count'] / flow_data['ack_count'] if flow_data['ack_count'] > 0 else 0
        forward_backward_ratio = flow_data['forward_packets'] / flow_data['backward_packets'] if flow_data['backward_packets'] > 0 else flow_data['forward_packets']

        # Build feature dictionary (all 35 features, no label)
        features = {
            'src_ip': flow_data['src_ip'],
            'dst_ip': flow_data['dst_ip'],
            'src_port': flow_data['src_port'],
            'dst_port': flow_data['dst_port'],
            'protocol': flow_data['protocol'],
            'duration': duration,
            'total_packets': flow_data['packets'],
            'total_bytes': flow_data['bytes'],
            'min_packet_size': min(packet_sizes) if packet_sizes else 0,
            'max_packet_size': max(packet_sizes) if packet_sizes else 0,
            'avg_packet_size': np.mean(packet_sizes) if packet_sizes else 0,
            'std_packet_size': np.std(packet_sizes) if packet_sizes else 0,
            'syn_count': flow_data['syn_count'],
            'ack_count': flow_data['ack_count'],
            'fin_count': flow_data['fin_count'],
            'rst_count': flow_data['rst_count'],
            'psh_count': flow_data['psh_count'],
            'urg_count': flow_data['urg_count'],
            'packets_per_second': packets_per_second,
            'bytes_per_second': bytes_per_second,
            'bytes_per_packet': bytes_per_packet,
            'forward_packets': flow_data['forward_packets'],
            'backward_packets': flow_data['backward_packets'],
            'forward_bytes': flow_data['forward_bytes'],
            'backward_bytes': flow_data['backward_bytes'],
            'forward_backward_ratio': forward_backward_ratio,
            'avg_iat': np.mean(iats) if iats else 0,
            'std_iat': np.std(iats) if iats else 0,
            'min_iat': min(iats) if iats else 0,
            'max_iat': max(iats) if iats else 0,
            'syn_ack_ratio': syn_ack_ratio,

            # Port indicators
            'is_port_22': 1 if (flow_data['dst_port'] == 22 or flow_data['src_port'] == 22) else 0,
            'is_port_6200': 1 if (flow_data['dst_port'] == 6200 or flow_data['src_port'] == 6200) else 0,
            'is_ftp_port': 1 if (flow_data['dst_port'] == 21 or flow_data['src_port'] == 21) else 0,
            'is_ftp_data_port': 1 if (flow_data['dst_port'] == 20 or flow_data['src_port'] == 20) else 0,
        }

        features_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(features_list)

    print(f"[+] Extracted {len(df)} flows with {len(df.columns)} features")

    return df


def save_uploaded_pcap(uploaded_file):
    """
    Save uploaded Streamlit file to temporary PCAP file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        str: Path to temporary PCAP file
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name
