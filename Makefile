USER:=$(shell id -u)
GROUP:=$(shell id -g)
PWD:=$(shell pwd)

load:
	mkdir -p ~/training && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v ~/training:/tmp/training --env-file ./.env train-image python3 /tmp/code/scripts/move_images_from_db_to_fs.py

train:
	mkdir -p ~/training && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v ~/training:/tmp/training --env-file .env train-image python3 /tmp/code/scripts/train.py