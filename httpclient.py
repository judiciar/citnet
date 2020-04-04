#!/usr/bin/env python3
# Evalueaza un fisier PDF folosind serverul citnetd
# Reteaua neuronala CitNet foloseste modelul EfficientNet
# (vezi https://arxiv.org/abs/1905.11946)

# FOLOSIRE
# httpclient.py <pathfile> [<robot> <idinstanta>]
# httpclient.py /mnt/cagl/Prelucrate/name.pdf
# httpclient.py C:\doc\Prelucrate\name.pdf doc 1

import requests
import argparse

HOST = "10.20.48.100"  # Hostname sau adresa IP a server-ului
PORT = "1102"  # Portul folosit de server

ap = argparse.ArgumentParser()
ap.add_argument("pathfile", help="calea catre fisierul PDF")
ap.add_argument(
    "robot", help="numele robotului (cit/doc - optional)", nargs="?", default=""
)
ap.add_argument("idinstanta", help="id-ul instantei (optional)", nargs="?", default="")
args = vars(ap.parse_args())

r = requests.get(
    "http://{}:{}/query".format(HOST, PORT),
    params={
        "robot": args["robot"],
        "idinstanta": args["idinstanta"],
        "pathfile": args["pathfile"],
    },
)
print(r.content.decode("utf-8"))
