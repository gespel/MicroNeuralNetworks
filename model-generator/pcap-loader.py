from scapy.utils import RawPcapReader

def read_pcap(file_path):
    for pkt_data, pkt_metadata in RawPcapReader(file_path):
        print(pkt_data, pkt_metadata)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pcap-loader.py <pcap_file>")
        sys.exit(1)

    pcap_file = sys.argv[1]
    read_pcap(pcap_file)