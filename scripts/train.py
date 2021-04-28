from tensorflow import keras
import tensorflow as tf
from datetime import datetime
import os
import pandas as pd
import logging
import numpy as np
import tensorflowjs as tfjs

batch_size = 32
img_height = 224
img_width = 224
epochs = 1

IMAGE_FILE_PATH = '/tmp/training/images'
DATASET_NAME = os.environ['DATASET_NAME']

TRAINING_RUN_NAME = os.environ['TRAINING_RUN_NAME']
HISTORY_FILE_PATH = '/tmp/training/run-history'
save_dir = os.path.join(HISTORY_FILE_PATH, TRAINING_RUN_NAME, '{}'.format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S")) )
os.makedirs(save_dir, exist_ok=True)

#check that dataset folder exists and assume that it contains all the files necessary
dataset_folder = os.path.join(IMAGE_FILE_PATH, DATASET_NAME)
if not os.path.isdir(dataset_folder):
    raise Exception('Dataset does not exist. Run load_images_to_disk job')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_folder,
  validation_split=0.2,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  dataset_folder,
  validation_split=0.2,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(img_height, img_width),
  batch_size=batch_size)

dataset = val_ds.take(3)
#print(list(dataset.as_numpy_iterator()))

#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

IMG_SHAPE = (img_height, img_width, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, 
    weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss= 'categorical_crossentropy',
  metrics=['accuracy', 'categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

#print(model.summary())

#print('Number of trainable variables = {}'.format(len(model.trainable_variables)))

history = model.fit(train_ds, epochs=epochs, validation_data=val_ds)

model.save(os.path.join(save_dir, 'model.h5'))

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = os.path.join(save_dir, 'history.json')
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = os.path.join(save_dir, 'history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)


#make predictions and final evaluation
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
#results = model.evaluate(val_ds)
#print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for validation set")
#predictions = model.predict(val_ds)
#print(predictions)
#predictions_df = pd.DataFrame(predictions) 
#predictions_csv_file = os.path.join(save_dir, 'predictions.csv')
#with open(predictions_csv_file, mode='w') as f:
#    predictions_df.to_csv(f)

js_dir = os.path.join(save_dir, 'js')
os.makedirs(js_dir, exist_ok=True)
tfjs.converters.save_keras_model(model, js_dir)


#repoint symlink for most_recent run
most_recent_run_path = os.path.join(HISTORY_FILE_PATH, 'most_recent')
if os.path.islink(most_recent_run_path):
  os.unlink(most_recent_run_path)
os.symlink(save_dir, most_recent_run_path)


