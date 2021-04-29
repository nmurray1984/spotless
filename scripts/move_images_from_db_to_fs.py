import psycopg2
import os
import sys
import json
import os
import shutil

DATABASE_URL = os.environ['DATABASE_URL']
IMAGE_FILE_PATH = '/tmp/training/images'
DATASET_NAME = 'nsfwv03'

dataset_folder_path = os.path.join(IMAGE_FILE_PATH, DATASET_NAME)
if os.path.isdir(dataset_folder_path):
   shutil.rmtree(dataset_folder_path)

cache_folder_path = os.path.join(IMAGE_FILE_PATH, 'cache')
os.makedirs(cache_folder_path, exist_ok=True)
print('Created directory' + cache_folder_path)


query = '''
SELECT
   c.id,
   --c.image_bytes,
   di.classification
FROM
   nsfw_server.dataset_image di
   inner join nsfw_server.contributed_image c on di.image_id = c.id
   inner join nsfw_server.dataset d on di.dataset_id = d.id
WHERE
   d.name = '{}'
'''.format(DATASET_NAME)

conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()
cursor.execute(query)

image_bytes_query = 'select image_bytes from nsfw_server.contributed_image where id={}'

for row in cursor:
   image_id, image_classification = row

   image_file_name = str(image_id) + '.jpg'
   cache_file_path = os.path.join(cache_folder_path, image_file_name)
   sym_folder_path = os.path.join(IMAGE_FILE_PATH, DATASET_NAME, image_classification)
   sym_file_path = os.path.join(sym_folder_path, image_file_name)
   
   #check if file exists, and if so, don't write it again
   if not os.path.isfile(cache_file_path):
      cursor2 = conn.cursor()
      cursor2.execute(image_bytes_query.format(image_id))
      image_bytes = cursor2.fetchone()[0]
      cursor2.close()
      with open(cache_file_path, 'wb') as file:
         file.write(image_bytes)

   if not os.path.isdir(sym_folder_path):
        os.makedirs(sym_folder_path, exist_ok=True)
        print('Created directory' + sym_folder_path)
   
   if not os.path.islink(sym_file_path):
      os.symlink(cache_file_path, sym_file_path)

conn.close()
