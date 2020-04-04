#!/usr/bin/env python3
# Script de testare pentru reteaua neurala EfficientNet

# FOLOSIRE
# testing.py -i dataset/testing -m saved_model.hdf5
# testing.py -d dataset/testing -m saved_model.hdf5
# python3 testing.py --img dataset/testing --model saved_model.hdf5

# importa pachetele necesare
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import img_to_array
    from keras.models import Sequential, load_model
    from keras.layers import Dense
    from keras.optimizers import Adam
    from efficientnet.keras import EfficientNetB0
from imutils import paths
from os.path import dirname, basename
import numpy as np
import argparse
import cv2
import sys

# prelucreaza argumentele din linia de comanda
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--img",
    "-d",
    "--dataset",
    required=True,
    help="calea catre directorul cu imaginile ce vor fi evaluate",
)
ap.add_argument("-m", "--model", required=True, help="calea catre reteaua neur(on)ala")
args = vars(ap.parse_args())

# incarca reteaua neur(on)ala
print("[INFO] incarc reteaua neuronala CitNet...")
model=load_model(args["model"])

# ia toate subdirectoarele cu imagini din directorul de intrare
imagePaths = list(paths.list_images(args["img"]))
if len(imagePaths) == 0:
    print("[ERROR] nu am gasit niciun fisier imagine")
    sys.exit(1)
contorerori = contortotal = 0

# prelucreaza imaginile gasite
print("[INFO] evaluez imaginile...")
for p in imagePaths:
    # incarca imaginea
    orig = cv2.imread(p, cv2.IMREAD_COLOR)
    # preproceseaza imaginea (dimensiunea trebuie sa fie 64x128 pixeli),
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
    contortotal += 1
    if label != basename(dirname(p)):
        print("{} - {} scor: {:.4f}".format(p, label, pred[0][np.argmax(pred[0])]))
        contorerori += 1
    elif abs(pred[0][0] - pred[0][1]) <= 0.4:
        print(
            "La limita: {} - {} scor: {:.4f}".format(
                p, label, pred[0][np.argmax(pred[0])]
            )
        )

# afiseaza rezumatul
print(
    "[INFO] Rezumat: {} gresite din {} total. {:.4%} corecte, {:.4%} erori".format(
        contorerori,
        contortotal,
        1 - contorerori / contortotal,
        contorerori / contortotal,
    )
)
