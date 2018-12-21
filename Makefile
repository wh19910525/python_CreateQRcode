
all:
	rm *.pyc *.png -rf
	python qrcode.py
	sz test01.png 

