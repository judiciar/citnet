#!/usr/bin/env python3
# ResNet-50 v2 (vezi https://arxiv.org/abs/1603.05027)

# FOLOSIRE
# trainresnet50.py -d[sau -i] dataset_folder -m modelname[.hdf5]
# python3 trainresnet50.py --dataset dataset_folder --model modelname[.hdf5]

# importa pachetele necesare
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential, load_model
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.callbacks import ModelCheckpoint, CSVLogger
    from keras.applications.resnet_v2 import ResNet50V2
from imutils import paths
import numpy as np
import argparse
import os
import sys

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
NUM_EPOCHS = 100
BS = 32

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

# initializeaza generatorul setului de training
trainGen = trainAug.flow_from_directory(
    TRAIN_PATH,
    class_mode="categorical",
    target_size=(128, 64),
    shuffle=True,
    batch_size=BS,
)

# initializeaza generatorul setului de validare
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(128, 64),
    shuffle=False,
    batch_size=BS,
)

# daca fisierul modelului exista deja, intreaba utilizatorul daca vrea sa
# continue antrenamentul. Daca nu exista, initializeaza reteaua neur(on)ala
if os.path.isfile(args["model"]):
    while True:
        choice = input(
            "Acest model exista deja. Vreti sa continuati antrenamentul modelului? (y/n): "
        )
        if choice in ("y", "Y"):
            model = load_model(args["model"])
            break
        elif choice in ("n", "N"):
            sys.exit(0)

else:
    # initializeaza si compileaza modelul ResNet-50 v2 (foloseste transfer learning)
    res_net = ResNet50V2(
        weights="imagenet", input_shape=(128, 64, 3), include_top=False, pooling="max"
    )
    model = Sequential()
    model.add(res_net)
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))
    model.compile(
        optimizer=Adam(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()

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
csv_logger = CSVLogger("history.csv", separator=",", append=True)
print(
    "[INFO] Modelul este salvat pe disc dupa fiecare epoca in formatul: nume_model-epoca-training_set_accuracy-validation_set_accuracy.hdf5"
)

# antreneaza (fit) reteaua neur(on)ala
H = model.fit_generator(
    trainGen,
    steps_per_epoch=np.ceil(totalTrain / BS),
    validation_data=valGen,
    validation_steps=np.ceil(totalVal / BS),
    epochs=NUM_EPOCHS,
    callbacks=[checkpointer, csv_logger],
)

print(
    "[INFO] Recomand testarea fisierului de model .hdf5 dorit pe setul de test cu testing.py"
)
