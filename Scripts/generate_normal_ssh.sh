#!/bin/bash

# Configuration
VICTIM_IP="192.168.113.130"
SSH_USER="sshtest"              # User you created
SSH_PASS="testpass123"          # Password you set
NUM_SESSIONS=100                 # Number of normal SSH sessions
INTERFACE="eth0"

echo "=========================================="
echo "Normal SSH Traffic Generator"
echo "=========================================="
echo "[*] Target: $VICTIM_IP"
echo "[*] SSH User: $SSH_USER"
echo "[*] Sessions: $NUM_SESSIONS"
echo "=========================================="

# Check if sshpass is installed (for password automation)
if ! command -v sshpass &> /dev/null; then
    echo "[*] Installing sshpass..."
    sudo apt install sshpass -y
fi

# Start packet capture
echo "[*] Starting packet capture..."
sudo tcpdump -i $INTERFACE host $VICTIM_IP and port 22 -w normal_ssh_traffic2.pcap &
TCPDUMP_PID=$!
echo "[+] Capture started (PID: $TCPDUMP_PID)"
sleep 2

# Generate normal SSH sessions
echo "[*] Generating normal SSH sessions..."

for i in $(seq 1 $NUM_SESSIONS); do
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "[*] Progress: $i/$NUM_SESSIONS sessions"
    fi
    
    # Vary SSH commands for diversity
    case $((i % 6)) in
        0)
            # Simple login and logout
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "exit" 2>/dev/null
            ;;
        1)
            # Check directory and list files
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "pwd; ls" 2>/dev/null
            ;;
        2)
            # System info commands
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "whoami; hostname" 2>/dev/null
            ;;
        3)
            # File operations
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "ls -la; pwd" 2>/dev/null
            ;;
        4)
            # Check processes
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "ps aux | head -10" 2>/dev/null
            ;;
        5)
            # Network info
            sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no $SSH_USER@$VICTIM_IP "ifconfig; netstat -an | head -5" 2>/dev/null
            ;;
    esac
    
    # Random delay 2-5 seconds (simulate human behavior)
    DELAY=$((2 + RANDOM % 4))
    sleep $DELAY
    
done

# Stop capture
echo "[*] Stopping capture..."
sudo kill $TCPDUMP_PID
sleep 2

echo "=========================================="
echo "[+] Done! Normal SSH traffic captured"
echo "[+] PCAP file: normal_ssh_traffic.pcap"
echo "[+] Sessions: $NUM_SESSIONS"
echo "=========================================="
