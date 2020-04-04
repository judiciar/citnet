#!/usr/bin/env python3
# Script de antrenament pentru reteaua neuronala CitNet
# CitNet foloseste modelul EfficientNet (vezi https://arxiv.org/abs/1905.11946)

# FOLOSIRE
# train.py -d[sau -i] dataset_folder <-m model.json> -w weights.hdf5
# train.py --dataset folder <--model model.json> --weights weights.hdf5

# importa pachetele necesare
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential, model_from_json
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, CSVLogger
    from efficientnet.keras import EfficientNetB0
from imutils import paths
import numpy as np
import argparse
import sys

# initializeaza numarul de epoci de antrenament si marimea lotului (batch size)
NUM_EPOCHS = 25
BS = 32

# prelucreaza argumentele din linia de comanda
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    "-i",
    "--img",
    type=str,
    required=True,
    help="calea catre dataset-ul cu imagini",
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

# compune calea catre directoarele in care se afla imaginile
# de antrenament si validare
TRAIN_PATH = os.path.sep.join([args["dataset"], "training"])
VAL_PATH = os.path.sep.join([args["dataset"], "validation"])

# determina numarul total de imagini din aceste directoare
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))

# initializeaza si augmenteaza setul de antrenament
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=0,
    vertical_flip=False,
    fill_mode="constant",
    cval=0,
)

# initializeaza setul de validare
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initializeaza generatorul setului de antrenament
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=(256, 128),
    shuffle=True,
    batch_size=BS,
)

# initializeaza generatorul setului de validare
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(256, 128),
    shuffle=False,
    batch_size=BS,
)

# daca fisierul modelului exista deja, intreaba utilizatorul daca vrea sa
# continue antrenamentul. Daca nu exista, initializeaza reteaua neur(on)ala
if os.path.isfile(args["model"]) and os.path.isfile(args["weights"]):
    while True:
        choice = input(
            "Acest model exista deja. Vreti sa continuati antrenamentul modelului? (y/n): "
        )
        if choice in ("y", "Y"):
            json_file = open(args["model"], "r")
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json)
            model.load_weights(args["weights"])
            model.compile(
                optimizer=Adam(lr=0.0001),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            break
        elif choice in ("n", "N"):
            sys.exit(0)

else:
    # initializeaza si compileaza modelul EfficientNet (foloseste transfer learning)
    efficient_net = EfficientNetB0(
        weights="imagenet", input_shape=(256, 128, 3), include_top=False, pooling="max"
    )
    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    model.compile(
        optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
    )
    if not os.path.isfile(args["model"]):
        model_json = model.to_json()
        with open(args["model"], "w") as json_file:
            json_file.write(model_json)
    model.summary()

# pregatiri pentru salvarea modelului (reteaua neuronala) pe disc
# dupa fiecare epoca de antrenament si notarea parametrilor
# fiecarei epoci intr-un fisier csv
filepath = (
    os.path.splitext(args["weights"])[0]
    + "-{epoch:02d}-acc{acc:.4f}-val_acc{val_acc:.4f}.hdf5"
)
checkpointer = ModelCheckpoint(
    filepath,
    monitor="val_acc",
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode="auto",
    period=1,
)
csv_logger = CSVLogger("history.csv", separator=",", append=True)
print(
    "[INFO] Modelul este salvat pe disc dupa fiecare epoca in formatul: nume_model-epoca-training_set_accuracy-validation_set_accuracy.hdf5"
)

# antreneaza (fit) reteaua neuronala
model.fit_generator(
    trainGen,
    steps_per_epoch=np.ceil(totalTrain / BS),
    validation_data=valGen,
    validation_steps=np.ceil(totalVal / BS),
    epochs=NUM_EPOCHS,
    callbacks=[checkpointer, csv_logger],
)

print(
    "[INFO] Recomand testarea fisierului de model .hdf5 dorit pe setul de test cu test.py"
)
