#!/bin/bash

# Configuration
VICTIM_IP="192.168.113.130"
FTP_USER="ftp"
FTP_PASS="ftp"
NUM_SESSIONS=50            # Generate enough sessions for 500+ flows
INTERFACE="eth0"

echo "========================================================"
echo "Normal FTP Traffic Generator - Target: 500+ Flows"
echo "========================================================"
echo "[*] Target: $VICTIM_IP"
echo "[*] Sessions to generate: $NUM_SESSIONS"
echo "[*] Estimated flows: 500-600"
echo "========================================================"

# Start packet capture
echo "[*] Starting packet capture..."
sudo tcpdump -i $INTERFACE host $VICTIM_IP and port 21 -w normal_traffic_100flows.pcap &
TCPDUMP_PID=$!
echo "[+] Capture started (PID: $TCPDUMP_PID)"
sleep 2

# Generate varied normal FTP sessions
for i in $(seq 1 $NUM_SESSIONS); do
    
    # Progress indicator
    if [ $((i % 10)) -eq 0 ]; then
        echo "[*] Progress: $i/$NUM_SESSIONS sessions completed"
    fi
    
    # Vary commands based on session number to create different flows
    case $((i % 5)) in
        0)
            # Simple listing
            ftp -inv $VICTIM_IP <<EOF
user $FTP_USER $FTP_PASS
ls
quit
EOF
            ;;
        1)
            # Directory navigation
            ftp -inv $VICTIM_IP <<EOF
user $FTP_USER $FTP_PASS
pwd
cd /
ls
quit
EOF
            ;;
        2)
            # Binary mode
            ftp -inv $VICTIM_IP <<EOF
user $FTP_USER $FTP_PASS
binary
dir
quit
EOF
            ;;
        3)
            # Multiple commands
            ftp -inv $VICTIM_IP <<EOF
user $FTP_USER $FTP_PASS
ls
pwd
dir
ls -la
quit
EOF
            ;;
        4)
            # Status check
            ftp -inv $VICTIM_IP <<EOF
user $FTP_USER $FTP_PASS
status
pwd
ls
quit
EOF
            ;;
    esac
    
    # Random delay between 2-5 seconds to prevent flow aggregation
    DELAY=$((2 + RANDOM % 4))
    sleep $DELAY
    
done

# Stop capture
echo ""
echo "[*] Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2

echo ""
echo "========================================================"
echo "[+] Traffic generation complete!"
echo "[+] PCAP file: normal_traffic_500flows.pcap"
echo "[+] Sessions generated: $NUM_SESSIONS"
echo ""
echo "[*] Next steps:"
echo "    1. Verify flow count: tshark -r normal_traffic_500flows.pcap -q -z conv,tcp"
echo "    2. Or run: python3 pcap_to_csv_enhanced.py"
echo "========================================================"
