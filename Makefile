
all:
	rm *.pyc *.png -rf
	python qrcode.py
	sz *.png

