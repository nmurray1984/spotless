import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import glob
import os

parser = argparse.ArgumentParser('Inference on trained model')

parser.add_argument('--model',
                    help='model trained on top of MobileNetV2')

args = parser.parse_args()

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

IMAGE_FILE_PATH = '/tmp/training/images'
DATASET_NAME = os.environ['DATASET_NAME']

#check that dataset folder exists and assume that it contains all the files necessary
dataset_folder = os.path.join(IMAGE_FILE_PATH, DATASET_NAME)
if not os.path.isdir(dataset_folder):
    raise Exception('Dataset does not exist. Run load_images_to_disk job')

ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_folder,
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

model = load_model(args.model, compile=False)
model.compile(
  optimizer='adam',
  loss= 'categorical_crossentropy',
  metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
results = model.evaluate(val_ds)
print("Results: ", results)


