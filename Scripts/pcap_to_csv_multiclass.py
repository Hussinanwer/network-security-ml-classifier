#!/usr/bin/env python3

import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict
import os

def extract_features_enhanced(pcap_file, label):
    """
    Extract comprehensive network features from PCAP using PyShark
    label: 0 = normal, 1 = vsftpd backdoor, 2 = SSH brute force
    """
    print(f"[*] Reading {pcap_file}...")
    
    if not os.path.exists(pcap_file):
        print(f"[!] Error: {pcap_file} not found!")
        return []
    
    # Read PCAP file
    cap = pyshark.FileCapture(pcap_file, keep_packets=False)
    
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
    for packet in cap:
        packet_count += 1
        
        try:
            if not hasattr(packet, 'ip'):
                continue
            
            src_ip = packet.ip.src
            dst_ip = packet.ip.dst
            
            if hasattr(packet, 'tcp'):
                protocol = 'TCP'
                src_port = int(packet.tcp.srcport)
                dst_port = int(packet.tcp.dstport)
                
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
                packet_length = int(packet.length)
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
                
                timestamp = float(packet.sniff_timestamp)
                flow['packet_times'].append(timestamp)
                if flow['start_time'] is None:
                    flow['start_time'] = timestamp
                flow['end_time'] = timestamp
                
                tcp_flags = int(packet.tcp.flags, 16)
                if tcp_flags & 0x02: flow['syn_count'] += 1
                if tcp_flags & 0x10: flow['ack_count'] += 1
                if tcp_flags & 0x01: flow['fin_count'] += 1
                if tcp_flags & 0x04: flow['rst_count'] += 1
                if tcp_flags & 0x08: flow['psh_count'] += 1
                if tcp_flags & 0x20: flow['urg_count'] += 1
                    
            elif hasattr(packet, 'udp'):
                protocol = 'UDP'
                src_port = int(packet.udp.srcport)
                dst_port = int(packet.udp.dstport)
                
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
                packet_length = int(packet.length)
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
                
                timestamp = float(packet.sniff_timestamp)
                flow['packet_times'].append(timestamp)
                if flow['start_time'] is None:
                    flow['start_time'] = timestamp
                flow['end_time'] = timestamp
                
        except (AttributeError, ValueError) as e:
            continue
    
    cap.close()
    print(f"[+] Processed {packet_count} packets, found {len(flows)} flows")
    
    # Convert flows to feature list
    features_list = []
    
    for flow_key, flow_data in flows.items():
        if flow_data['packets'] == 0:
            continue
        
        duration = 0
        if flow_data['start_time'] and flow_data['end_time']:
            duration = flow_data['end_time'] - flow_data['start_time']
        
        packet_sizes = flow_data['packet_sizes']
        
        packet_times = sorted(flow_data['packet_times'])
        iats = []
        if len(packet_times) > 1:
            for i in range(1, len(packet_times)):
                iats.append(packet_times[i] - packet_times[i-1])
        
        packets_per_second = flow_data['packets'] / duration if duration > 0 else 0
        bytes_per_second = flow_data['bytes'] / duration if duration > 0 else 0
        bytes_per_packet = flow_data['bytes'] / flow_data['packets'] if flow_data['packets'] > 0 else 0
        
        syn_ack_ratio = flow_data['syn_count'] / flow_data['ack_count'] if flow_data['ack_count'] > 0 else 0
        forward_backward_ratio = flow_data['forward_packets'] / flow_data['backward_packets'] if flow_data['backward_packets'] > 0 else flow_data['forward_packets']
        
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
            
            # Attack-specific indicators
            'is_port_22': 1 if (flow_data['dst_port'] == 22 or flow_data['src_port'] == 22) else 0,
            'is_port_6200': 1 if (flow_data['dst_port'] == 6200 or flow_data['src_port'] == 6200) else 0,
            'is_ftp_port': 1 if (flow_data['dst_port'] == 21 or flow_data['src_port'] == 21) else 0,
            'is_ftp_data_port': 1 if (flow_data['dst_port'] == 20 or flow_data['src_port'] == 20) else 0,
            
            'label': label
        }
        
        features_list.append(features)
    
    return features_list

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("Multi-Class PCAP to CSV Feature Extraction")
    print("=" * 70)
    
    # Extract normal traffic - FTP
    print("\n[*] Processing normal FTP traffic (500 flows)...")
    normal_ftp_500 = extract_features_enhanced('normal_traffic_500flows.pcap', label=0)
    print(f"[+] Extracted {len(normal_ftp_500)} normal FTP flows")
    
    print("\n[*] Processing normal FTP traffic (100 flows)...")
    normal_ftp_100 = extract_features_enhanced('normal_traffic_100flows.pcap', label=0)
    print(f"[+] Extracted {len(normal_ftp_100)} normal FTP flows")
    
    # Extract normal traffic - SSH
    print("\n[*] Processing normal SSH traffic (file 1)...")
    normal_ssh = extract_features_enhanced('normal_ssh_traffic.pcap', label=0)
    print(f"[+] Extracted {len(normal_ssh)} normal SSH flows")
    
    print("\n[*] Processing normal SSH traffic (file 2)...")
    normal_ssh1 = extract_features_enhanced('normal_ssh_traffic1.pcap', label=0)
    print(f"[+] Extracted {len(normal_ssh1)} normal SSH flows")
    
    print("\n[*] Processing normal SSH traffic (file 3)...")
    normal_ssh2 = extract_features_enhanced('normal_ssh_traffic2.pcap', label=0)
    print(f"[+] Extracted {len(normal_ssh2)} normal SSH flows")
    
    # Extract vsftpd backdoor attack
    print("\n[*] Processing vsftpd backdoor attack...")
    vsftpd_features = extract_features_enhanced('attack_traffic.pcap', label=1)
    print(f"[+] Extracted {len(vsftpd_features)} vsftpd backdoor flows")
    
    # Extract SSH brute force attack
    print("\n[*] Processing SSH brute force attack...")
    ssh_bruteforce = extract_features_enhanced('attack_ssh_traffic.pcap', label=2)
    print(f"[+] Extracted {len(ssh_bruteforce)} SSH brute force flows")
    
    # Combine ALL datasets
    print("\n[*] Combining datasets...")
    all_features = normal_ftp_500 + normal_ftp_100 + normal_ssh + normal_ssh1 + normal_ssh2 + vsftpd_features + ssh_bruteforce
    
    print(f"\n[+] Dataset Summary (Before Balancing):")
    print(f"    - Normal FTP flows: {len(normal_ftp_500) + len(normal_ftp_100)}")
    print(f"    - Normal SSH flows: {len(normal_ssh) + len(normal_ssh1) + len(normal_ssh2)}")
    print(f"    - Total Normal flows: {len(normal_ftp_500) + len(normal_ftp_100) + len(normal_ssh) + len(normal_ssh1) + len(normal_ssh2)}")
    print(f"    - vsftpd backdoor flows: {len(vsftpd_features)}")
    print(f"    - SSH brute force flows: {len(ssh_bruteforce)}")
    print(f"    - Total flows: {len(all_features)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # ============================================================
    # DOWNSAMPLE LABEL 2 (SSH Brute Force) FOR BALANCE
    # ============================================================
    print("\n" + "=" * 70)
    print("BALANCING DATASET - DOWNSAMPLING LABEL 2")
    print("=" * 70)
    
    # Current counts
    label_0_count = sum(df['label'] == 0)
    label_1_count = sum(df['label'] == 1)
    label_2_count = sum(df['label'] == 2)
    
    print(f"\n[*] Before downsampling:")
    print(f"    - Label 0 (Normal):      {label_0_count}")
    print(f"    - Label 1 (vsftpd):      {label_1_count}")
    print(f"    - Label 2 (SSH BF):      {label_2_count}")
    
    # Calculate target for label 2 (average of label 0 and 1)
    target_label_2 = int((label_0_count + label_1_count) / 2)
    
    # Optional: Set specific target manually
    # target_label_2 = 450  # Uncomment and set your desired number
    
    print(f"\n[*] Target for label 2: {target_label_2}")
    
    # Downsample if needed
    if label_2_count > target_label_2:
        print(f"[*] Downsampling label 2 from {label_2_count} to {target_label_2}...")
        
        # Separate label 2 from others
        df_label_2 = df[df['label'] == 2]
        df_others = df[df['label'] != 2]
        
        # Randomly sample label 2 to target size
        df_label_2_sampled = df_label_2.sample(n=target_label_2, random_state=42)
        
        # Combine back
        df = pd.concat([df_others, df_label_2_sampled], ignore_index=True)
        
        print(f"\n[+] After downsampling:")
        print(f"    - Label 0 (Normal):      {sum(df['label'] == 0)}")
        print(f"    - Label 1 (vsftpd):      {sum(df['label'] == 1)}")
        print(f"    - Label 2 (SSH BF):      {sum(df['label'] == 2)}")
        print(f"    - Reduced by:            {label_2_count - target_label_2} flows")
        print(f"    - New total:             {len(df)} flows")
    else:
        print(f"[+] No downsampling needed - label 2 already at or below target")
    
    print("=" * 70)
    
    # Shuffle dataset
    print("\n[*] Shuffling dataset...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("[+] Dataset shuffled!")
    
    # Save to CSV
    output_file = 'network_traffic_multiclass_dataset.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"[+] Dataset saved to: {output_file}")
    print(f"[+] Total records: {len(df)}")
    
    print(f"\n[+] Final Label Distribution:")
    print(f"    - Label 0 (Normal):      {sum(df['label'] == 0)}")
    print(f"    - Label 1 (vsftpd):      {sum(df['label'] == 1)}")
    print(f"    - Label 2 (SSH BF):      {sum(df['label'] == 2)}")
    
    print(f"\n[+] Total features: {len(df.columns) - 1}")
    
    print(f"\n[+] Port indicators:")
    print(f"    - Port 22 (SSH):         {df['is_port_22'].sum()} flows")
    print(f"    - Port 6200 (vsftpd):    {df['is_port_6200'].sum()} flows")
    print(f"    - Port 21 (FTP):         {df['is_ftp_port'].sum()} flows")
    
    print("\n" + "=" * 70)
    print("[+] MULTI-CLASS DATASET COMPLETE!")
    print("[+] Dataset is balanced and ready for ML!")
    print("=" * 70)
