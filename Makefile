USER:=$(shell id -u)
GROUP:=$(shell id -g)
PWD:=$(shell pwd)

load:
	mkdir -p ~/training && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v ~/training:/tmp/training:rw --env-file ./.env train-image python3 /tmp/code/scripts/move_images_from_db_to_fs.py

train:
	mkdir -p ~/training/run-history && \
	cat .env >> ~/training/run-history/stdouterr.txt && \
	docker build -t train-image . && \
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp/code -v ~/training:/tmp/training:rw --env-file .env train-image python3 /tmp/code/scripts/train.py 2>&1 | tee -a ~/training/run-history/stdouterr.txt && \
	mv ~/training/run-history/stdouterr.txt ~/training/run-history/most_recent/stdouterr.txt && \
	cp $(PWD)/scripts/train.py ~/training/run-history/most_recent/train-archive.py