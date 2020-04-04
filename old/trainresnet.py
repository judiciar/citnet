#!/usr/bin/env python3
# ResNet (Residual Neural Network)
# vezi https://arxiv.org/abs/1512.03385

# FOLOSIRE
# trainresnet.py -d[sau -i] dataset_folder -m modelname[.hdf5]
# python3 trainresnet.py --dataset dataset_folder --model modelname[.hdf5]

# importa pachetele necesare
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from keras.preprocessing.image import ImageDataGenerator
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint, CSVLogger
    from keras.models import load_model
    from resnet.resnet import ResNet
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
    color_mode="grayscale",
    shuffle=True,
    batch_size=BS,
)

# initializeaza generatorul setului de validare
valGen = valAug.flow_from_directory(
    VAL_PATH,
    class_mode="categorical",
    target_size=(128, 64),
    color_mode="grayscale",
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
    # initializeaza si compileaza implementarea Keras a modelului ResNet
    model = ResNet.build(64, 128, 1, 2, (2, 2, 3), (32, 64, 128, 256), reg=0.0005)
    opt = SGD(lr=1e-1, momentum=0.9, decay=1e-1 / NUM_EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
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
