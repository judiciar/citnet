#!/usr/bin/env python3
# Script de evaluare (predictie) cu reteaua neuronala CitNet
# a fisierelor PDF sau directoarelor indicate de utilizator
# CitNet foloseste modelul EfficientNet (vezi https://arxiv.org/abs/1905.11946)

# FOLOSIRE
# predict.py -i /mnt/br/JdBraila/Erori -m <model.json> -w weights.hdf5
# predict.py --img filename.pdf <--model model.json> --weights weights.hdf5

# importa pachetele necesare
from os import environ
from os.path import basename, splitext, isdir, isfile
environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import img_to_array
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense
    from keras.optimizers import Adam
    from efficientnet.keras import EfficientNetB0
from imutils import paths
import tempfile
import numpy as np
import argparse
import cv2
import sys
import subprocess

# prelucreaza argumentele din linia de comanda
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--img",
    "-d",
    "--dataset",
    required=True,
    help="calea catre fisierul PDF sau catre directorul cu fisiere PDF",
)
ap.add_argument(
    "-m",
    "--model",
    type=str,
    default="citnet_model.json",
    help="numele fisierul in care se salveaza modelul",
)
ap.add_argument(
    "-w",
    "--weights",
    type=str,
    required=True,
    help="fisierul cu greutatile retelei neur(on)ale",
)
args = vars(ap.parse_args())

# incarca reteaua neur(on)ala
print("[INFO] incarc reteaua neuronala CitNet...")
json_file = open(args["model"], "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(args["weights"])

# ia fisiere PDF si le converteste intr-un director temporar
print("[INFO] convertesc fisierele PDF in JPG...")
with tempfile.TemporaryDirectory() as directory:
    if isdir(args["img"]):
        p = subprocess.Popen(
            'find "'
            + args["img"]
            + '" -type f -name "*.pdf" | parallel -j50% "convert xc:white[128x256\!] -respect-parenthesis \( {}[0-1] -resize 595x842\> -background white -extent 595x842 -resize 128x128! -append \) -composite -set colorspace Gray -depth 8 -strip '
            + directory
            + '/{/.}.jpg"',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    elif isfile(args["img"]):
        p = subprocess.Popen(
            'convert xc:white[128x256\!] -respect-parenthesis \( "'
            + args["img"]
            + '"[0-1] -resize 595x842\> -background white -extent 595x842 -resize 128x128! -append \) -composite -set colorspace Gray -depth 8 -strip "'
            + directory
            + "/"
            + splitext(basename(args["img"]))[0]
            + '.jpg"',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    else:
        print("[ERROR] Nu exista fisierul sau folderul", args["img"])
        sys.exit(2)
    for line in p.stdout.readlines():
        print(line.decode("utf-8"))
    p.wait()
    imagePaths = list(paths.list_images(directory))
    if len(imagePaths) == 0:
        print("[ERROR] nu am gasit niciun fisier PDF")
        sys.exit(1)
    contortotal = 0

    # prelucreaza imaginile din directorul temporar
    print("[INFO] evaluez imaginile...")
    for p in imagePaths:
        # incarca imaginea
        orig = cv2.imread(p, cv2.IMREAD_COLOR)
        # preproceseaza imaginea (dimensiunea trebuie sa fie 128x256 pixeli),
        # intensitatea pixelilor urmand a fi redimensionata in intervalul [0, 1]
        image = orig.astype("float") / 255.0
        # image = cv2.resize(image, (64, 128))
        # ordoneaza canalele imaginii (canalele la inceput sau canalele la sfarsit)
        # in functie de Keras backend, dupa care adauga o dimensiune batch imaginii
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # evalueaza imaginea cu ajutorul retelei neur(on)ale
        pred = model.predict(image)
        # pred[0][0] este probabilitatea de 'citatii',
        # pred[0][1] este probabilitatea de 'documente'
        label = "documente" if pred[0][0] <= 0.5 else "citatii"
        print(
            "{}.pdf - {} scor: {:.4f}".format(
                splitext(basename(p))[0], label, pred[0][np.argmax(pred[0])]
            )
        )
        contortotal += 1
print("[INFO] Rezumat: {} total.".format(contortotal))
