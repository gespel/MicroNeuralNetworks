from scapy.all import rdpcap, IP, TCP, UDP, ICMP, Raw
import pandas as pd
import numpy as np
from collections import defaultdict

def extract_features(packets):
    """
    Extrahiert IDS-relevante Features aus Netzwerkpaketen.
    
    Features pro Paket:
    - packet_size: Paketgröße in Bytes
    - ttl: Time-To-Live
    - flags: TCP-Flags (als Integer)
    - src_port: Quellport (0 wenn nicht TCP/UDP)
    - dst_port: Zielport (0 wenn nicht TCP/UDP)
    - protocol: Protokoll (6=TCP, 17=UDP, 1=ICMP, 0=andere)
    - payload_size: Payload-Größe
    - tcp_window: TCP-Fenster-Größe (0 wenn nicht TCP)
    """
    features_list = []
    
    for packet in packets:
        features = {}
        
        # Paketgröße
        features['packet_size'] = len(packet)
        
        # IP-Layer
        if IP in packet:
            ip_layer = packet[IP]
            features['ttl'] = ip_layer.ttl
            features['protocol'] = ip_layer.proto
        else:
            features['ttl'] = 0
            features['protocol'] = 0
        
        # TCP-Layer
        if TCP in packet:
            tcp_layer = packet[TCP]
            features['src_port'] = tcp_layer.sport
            features['dst_port'] = tcp_layer.dport
            features['flags'] = int(tcp_layer.flags)
            features['tcp_window'] = tcp_layer.window
        else:
            features['src_port'] = 0
            features['dst_port'] = 0
            features['flags'] = 0
            features['tcp_window'] = 0
        
        # UDP-Layer
        if UDP in packet:
            udp_layer = packet[UDP]
            features['src_port'] = udp_layer.sport
            features['dst_port'] = udp_layer.dport
        
        # Payload-Größe
        if Raw in packet:
            features['payload_size'] = len(packet[Raw].load)
        else:
            features['payload_size'] = 0
        
        features_list.append(features)
    
    return features_list


def features_to_dataframe(features_list, normalize=True):
    """
    Konvertiert Feature-Liste zu Pandas DataFrame.
    
    Spalten: [packet_size, ttl, protocol, src_port, dst_port, flags, payload_size, tcp_window]
    """
    df = pd.DataFrame(features_list)
    
    if normalize:
        # Min-Max Normalisierung auf [0, 1]
        df = normalize_features_df(df)
    
    return df


def normalize_features_df(df):
    """Normalisiert Features auf [0, 1] (Min-Max Normalisierung)."""
    df_normalized = df.copy()
    
    for col in df.columns:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        
        if range_val == 0:
            range_val = 1
        
        df_normalized[col] = (df[col] - min_val) / range_val
    
    return df_normalized


def read_pcap(file_path, verbose=False):
    """Liest PCAP-Datei und extrahiert Features."""
    try:
        packets = rdpcap(file_path)
        print(f"[*] {len(packets)} Pakete geladen")
        
        features_list = extract_features(packets)
        df = features_to_dataframe(features_list, normalize=True)
        
        if verbose:
            print(f"\n[*] DataFrame Shape: {df.shape}")
            print(f"[*] Spalten: {list(df.columns)}")
            print(f"\n[*] Erste 5 Pakete (normalisiert):")
            print(df.head())
            print(f"\n[*] Statistiken:")
            print(df.describe())
        
        return df
    
    except Exception as e:
        print(f"[!] Fehler beim Laden der PCAP-Datei: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pcap-loader.py <pcap_file> [-v]")
        sys.exit(1)

    pcap_file = sys.argv[1]
    verbose = '-v' in sys.argv
    
    df = read_pcap(pcap_file, verbose=verbose)
    
    if df is not None:
        # Speichere Features als CSV
        csv_file = 'features.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n[+] Features gespeichert in '{csv_file}'")