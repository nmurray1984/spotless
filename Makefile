USER:=$(shell id -u)
GROUP:=$(shell id -g)
PWD:=$(shell pwd)

run:
	docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp tensorflow/tensorflow

exec:
	docker build -t train-image . && docker run -u $(USER):$(GROUP) -it -v $(PWD):/tmp train-image python3 /tmp/scripts/train.py