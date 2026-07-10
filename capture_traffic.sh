#!/usr/bin/env bash
set -euo pipefail

# Netzwerk-Traffic mitschneiden und als PCAP speichern

INTERFACE=""
DURATION=""
COUNT=""
FILTER=""
OUTPUT_DIR="."

usage() {
    cat <<EOF
Usage: $0 -i INTERFACE [OPTIONS]

Optionen:
  -i INTERFACE   Netzwerkinterface (z.B. eth0, wlan0, lo)
  -d SEKUNDEN    Aufnahmedauer (optional)
  -c ANZAHL      Anzahl Pakete (optional)
  -f FILTER      tcpdump-Filterausdruck (optional, z.B. 'port 80')
  -o VERZEICHNIS Ausgabeverzeichnis (default: $OUTPUT_DIR)
  -h             Hilfe anzeigen

Beispiel:
  sudo $0 -i eth0 -d 60 -f "port 443"
EOF
}

while getopts "i:d:c:f:o:h" opt; do
    case "$opt" in
        i) INTERFACE="$OPTARG" ;;
        d) DURATION="$OPTARG" ;;
        c) COUNT="$OPTARG" ;;
        f) FILTER="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

if [[ -z "$INTERFACE" ]]; then
    echo "Fehler: Interface (-i) ist erforderlich." >&2
    usage
    exit 1
fi

if [[ "$EUID" -ne 0 ]]; then
    echo "Hinweis: tcpdump benötigt meist Root-Rechte. Starte mit sudo ..." >&2
fi

if ! command -v tcpdump &>/dev/null; then
    echo "Fehler: tcpdump ist nicht installiert." >&2
    echo "Installiere es z.B. mit: sudo apt install tcpdump" >&2
    exit 1
fi

if ! ip link show "$INTERFACE" &>/dev/null; then
    echo "Fehler: Interface '$INTERFACE' nicht gefunden. Verfügbare Interfaces:" >&2
    ip -brief link show | awk '{print "  " $1}' >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$OUTPUT_DIR/capture_${INTERFACE}_${TIMESTAMP}.pcap"

TCPDUMP_ARGS=(-i "$INTERFACE" -w "$OUTPUT_FILE")
[[ -n "$COUNT" ]] && TCPDUMP_ARGS+=(-c "$COUNT")
[[ -n "$DURATION" ]] && TCPDUMP_ARGS+=("${FILTER:-}")

# Filter separat anhängen, damit er nicht als Interface-Argument interpretiert wird
if [[ -n "$FILTER" ]]; then
    TCPDUMP_ARGS+=("$FILTER")
fi

echo "Starte Mitschnitt auf '$INTERFACE' ..."
echo "Ausgabedatei: $OUTPUT_FILE"
[[ -n "$DURATION" ]] && echo "Dauer: $DURATION Sekunden"
[[ -n "$COUNT" ]] && echo "Max. Pakete: $COUNT"
[[ -n "$FILTER" ]] && echo "Filter: $FILTER"

if [[ -n "$DURATION" ]]; then
    timeout "$DURATION" tcpdump "${TCPDUMP_ARGS[@]}" || true
else
    tcpdump "${TCPDUMP_ARGS[@]}"
fi

echo "Mitschnitt beendet: $OUTPUT_FILE"
ls -lh "$OUTPUT_FILE"
