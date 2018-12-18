# qrcode.py
#coding:utf-8

from PIL import Image, ImageDraw
from numpy import *;

aa = 'abbdbdbbf'

print "----------- start, %r -----------" % aa.count('b')

'''
参数:
    bitmap        :QR码矩阵
    qrcodesize    :图像宽度, 单位:像素
    filename      :保存文件名
'''
def _genImage(bitmap, qrcodesize, filename):
    width = qrcodesize
    height = qrcodesize
    '''
    Generate image corresponding to the input bitmap
    with specified qrcodesize and filename.
    '''
    # New image in black-white mode initialized with white.
    img = Image.new('1', (width, height), 'white')
    drw = ImageDraw.Draw(img)

    # Normalized pixel width.
    print "bitmap-len=%r" % (len(bitmap))
    print "bitmap-len=%r" % (bitmap)

    a_unit_size = qrcodesize / len(bitmap)
    print "a-rectangle-size=%r" % (a_unit_size)

    for y in range(width):
        # Normalized y coordinate in bitmap
        normalj = y / a_unit_size

        for x in range(height):
            # Normalized x coordinate in bitmap
            normali = x / a_unit_size

            if normalj < len(bitmap) and normali < len(bitmap):
                # Draw pixel.
                drw.point((x, y), fill=bitmap[normalj][normali])

    img.save(filename)

print "------------ end, %r ------------" % aa.count('b')

test = [[ (i+j)%2 for i in range(8) ] for j in range(8)]

_genImage(test, 240, 'test.png')

