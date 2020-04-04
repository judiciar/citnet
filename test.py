#!/usr/bin/env python3
# Script de testare pentru reteaua neuronala CitNet
# CitNet foloseste modelul EfficientNet (vezi https://arxiv.org/abs/1905.11946)


# FOLOSIRE
# test.py -i dataset/testing -m <model.json> -w weights.hdf5
# python3 test.py -d dataset/testing <-m model.json> -w weights.hdf5
# test.py --img dataset/testing <--model model.json> --weights weights.hdf5

# importa pachetele necesare
from os import environ
from os.path import dirname, basename
environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint
    from efficientnet.keras import EfficientNetB0
from imutils import paths
import numpy as np
import pandas as pd
import argparse
import sys

# initializeaza marimea lotului (batch size) si alte variabile
BS = 32
contorerori = 0
categorii = []

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
ap.add_argument(
    "-m",
    "--model",
    default="citnet_model.json",
    help="fisierul cu modelul retelei neur(on)ale",
)
ap.add_argument(
    "-w", "--weights", required=True, help="fisierul cu greutatile retelei neur(on)ale",
)
args = vars(ap.parse_args())

# prelucreaza toate subdirectoarele cu imagini din directorul de intrare
filenames = list(paths.list_images(args["img"]))
contortotal = len(filenames)
if contortotal == 0:
    print("[ERROR] nu am gasit niciun fisier imagine")
    sys.exit(1)

# incarca reteaua neur(on)ala
print("[INFO] incarc reteaua neuronala CitNet...")
json_file = open(args["model"], "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(args["weights"])

# incarca numele fisierelor si categoria (label)
# intr-un obiect Pandas DataFrame
print("[INFO] incarc si evaluez imaginile...")
for filename in filenames:
    if basename(dirname(filename)) == "documente":
        categorii.append((0, 1))
    else:
        categorii.append((1, 0))

df = pd.DataFrame({"filename": filenames, "label": categorii})

# initializeaza setul de test
testAug = ImageDataGenerator(rescale=1 / 255.0)

# initializeaza generatorul setului de test
testGen = testAug.flow_from_dataframe(
    df,
    x_col="filename",
    y_col="label",
    class_mode="categorical",
    target_size=(256, 128),
    shuffle=False,
    batch_size=BS,
)

# evalueaza imaginea cu ajutorul retelei neuronale
predictii = model.predict_generator(testGen, steps=np.ceil(contortotal / BS), verbose=1)
rezultate = tuple(zip(testGen.filenames, predictii))

# verifica si afiseaza corectitudinea evaluarilor
for filename, pred in rezultate:
    label = "documente" if pred[0] <= 0.5 else "citatii"
    if label != basename(dirname(filename)):
        print("{} - {} scor: {:.4f}".format(filename, label, max(pred)))
        contorerori += 1
    elif abs(pred[0] - pred[1]) <= 0.4:
        print("La limita: {} - {} scor: {:.4f}".format(filename, label, max(pred)))

# afiseaza rezumatul
print(
    "[INFO] Rezumat: {} gresite din {} total. {:.4%} corecte, {:.4%} erori".format(
        contorerori,
        contortotal,
        1 - contorerori / contortotal,
        contorerori / contortotal,
    )
)
