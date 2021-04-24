USER:=$(shell id -u)
GROUP:=$(shell id -g)
PWD:=$(shell pwd)

load:
	mkdir -p ~/training && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v /tmp/training:/tmp/training:rw --env-file ./.env train-image python3 /tmp/code/scripts/move_images_from_db_to_fs.py

train:
	mkdir -p ~/training && \
	cat .env >> /tmp/training/run-history/stdouterr.txt && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v /tmp/training:/tmp/training:rw --env-file .env train-image python3 /tmp/code/scripts/train.py 2>&1 | tee -a /tmp/training/run-history/stdouterr.txt && \
	mv /tmp/training/run-history/stdouterr.txt /tmp/training/run-history/most_recent/stdouterr.txt && \
	cp $(PWD)/scripts/train.py /tmp/training/run-history/most_recent/train-archive.py
	
