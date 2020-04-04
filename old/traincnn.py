#!/usr/bin/env python3
# Convolutional Neural Network (CNN)

# FOLOSIRE
# traincnn.py -d[sau -i] dataset_folder -m modelname[.hdf5]
# python3 traincnn.py --dataset dataset_folder --model modelname[.hdf5]

# importa pachetele necesare
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.models import Sequential, load_model
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np
from imutils import paths
import argparse
import os
import sys


def initialize_cnn():
    # Initializeaza CNN
    classifier = Sequential()

    # Pas 1 - Convolutie
    classifier.add(
        Conv2D(
            32,
            (3, 3),
            input_shape=(128, 64, 1),
            data_format="channels_last",
            activation="relu",
        )
    )

    # Pas 2 - Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(Dropout(0.2))

    # Adauga un al doilea strat de convolutie
    classifier.add(Conv2D(32, (3, 3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # classifier.add(Dropout(0.2))

    # Pas 3 - Flattening
    classifier.add(Flatten())

    # Pas 4 - Straturi complet conectate (Full connection)
    classifier.add(Dense(units=128, activation="relu"))
    # classifier.add(Dropout(0.5))
    classifier.add(Dense(units=2, activation="softmax"))

    # Compileaza CNN
    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    classifier.summary()

    return classifier


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
    required=True,
    help="numele fisierul in care se salveaza modelul",
)
args = vars(ap.parse_args())

# initializeaza numarul de epoci de antrenament si marimea lotului (batch size)
NUM_EPOCHS = 25
BS = 32

# compune calea catre directoarele in care se afla imaginile
# de antrenament si validare
# TEST_PATH = os.path.sep.join([args["dataset"], "testing"])
TRAIN_PATH = os.path.sep.join([args["dataset"], "training"])
VAL_PATH = os.path.sep.join([args["dataset"], "validation"])

# determina numarul total de imagini din aceste directoare
# totalTest = len(list(paths.list_images(TEST_PATH)))
totalTrain = len(list(paths.list_images(TRAIN_PATH)))
totalVal = len(list(paths.list_images(VAL_PATH)))

# genereaza seturile de antrenament si validare si
# augmenteaza setul de antrenament
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=0,
    vertical_flip=False,
    fill_mode="constant",
    cval=0,
)

valtest_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(128, 64),
    batch_size=BS,
    color_mode="grayscale",
    class_mode="categorical",
    classes=["citatii", "documente"],
)

val_set = valtest_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(128, 64),
    batch_size=BS,
    color_mode="grayscale",
    class_mode="categorical",
    classes=["citatii", "documente"],
)

# pregatiri pentru salvarea modelului (reteaua neurala) pe disc
# dupa fiecare epoca de antrenament si notarea parametrilor
# fiecarei epoci intr-un fisier csv
filepath = (
    os.path.basename(args["model"])
    + "-{epoch:02d}-acc{acc:.4f}-val_acc{val_acc:.4f}.hdf5"
)
checkpointer = ModelCheckpoint(
    filepath,
    monitor="val_acc",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    period=1,
)
csv_logger = CSVLogger("historycnn.csv", separator=",", append=True)
print(
    "[INFO] Modelul este salvat pe disc dupa fiecare epoca in formatul: nume_model-epoca-training_set_accuracy-validation_set_accuracy.hdf5"
)

# daca fisierul modelului exista deja, intreaba utilizatorul daca vrea sa
# continue antrenamentul. Daca nu exista, initializeaza reteaua neur(on)ala
if os.path.isfile(args["model"]):
    while True:
        choice = input(
            "Acest model exista deja. Vreti sa continuati antrenamentul modelului? (y/n): "
        )
        if choice in ("y", "Y"):
            classifier = load_model(args["model"])
            break
        elif choice in ("n", "N"):
            sys.exit(0)
else:
    classifier = initialize_cnn()

# antreneaza (fit) reteaua neurala
classifier.fit_generator(
    training_set,
    steps_per_epoch=np.ceil(totalTrain / BS),
    epochs=NUM_EPOCHS,
    validation_data=val_set,
    validation_steps=np.ceil(totalVal / BS),
    callbacks=[checkpointer, csv_logger],
)

print(
    "[INFO] Recomand testarea fisierului de model .hdf5 dorit pe setul de test cu testing.py"
)
