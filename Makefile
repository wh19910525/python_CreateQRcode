
all:
	@rm *.pyc *.png -rf
	@python createQRcode.py
	@touch tmp.png
	@sz *.png

clean:
	rm *.pyc *.png -rf

