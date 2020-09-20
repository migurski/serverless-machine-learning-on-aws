all: demo/inference.zip

demo/inference.zip: demo/inference.py
	zip -j demo/inference.zip demo/inference.py