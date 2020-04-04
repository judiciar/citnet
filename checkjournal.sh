#!/bin/bash
# Cauta posibile erori de evaluare a fisierelor PDF in jurnalul (log) daemon-ului (serviciului) citnetd
if [ -z "$1" ]; then
  timestamp="today"
else
  timestamp="$1"
fi
echo "Se verifica fisierele prelucrate de CitNet dupa criteriul: since $timestamp"
echo "Nu au regex de document si sunt etichetate ca citatii (probabil ok):"
journalctl -u citnetd --since "$timestamp" | grep citatii | grep -P -v ".*[0-9]+-(?!1([0-2])-)[1-9][0-9]*-." | sort -t"]" -u -k2,2
echo -e "\nNu au regex de document si sunt etichetate ca documente (de verificat):"
journalctl -u citnetd --since "$timestamp" | grep documente | grep -P -v ".*[0-9]+-(?!1([0-2])-)[1-9][0-9]*-." | sort -t"]" -u -k2,2
echo -e "\nAu regex de document si sunt etichetate ca citatii (posibile erori ale CitNet):"
journalctl -u citnetd --since "$timestamp" | grep citatii | grep -P ".*[0-9]+-(?!1([0-2])-)[1-9][0-9]*-." | sort -t"]" -u -k2,2