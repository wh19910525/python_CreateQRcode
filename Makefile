
all:
	@rm *.pyc *.png -rf
	@python qrcode.py
	@touch tmp.png
	@sz *.png

clean:
	rm *.pyc *.png -rf

