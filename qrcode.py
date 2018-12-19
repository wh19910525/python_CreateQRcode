# qrcode.py
#coding:utf-8

from PIL import Image, ImageDraw
from numpy import *;

aa = 'abbdbdbbf'

print "----------- start, %r -----------" % aa.count('b')

class CapacityOverflowException(Exception):
    '''Exception for data larger than 17 characters in V1-L byte mode.'''
    def __init__(self, arg):
        self.arg = arg

    def __str__(self):
        return repr(self.arg)

'''
创建 QR码 图像

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

    #用图像宽度 除以 矩阵维度得到 QR码中一个单位对应的像素数
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

'''
编码格式信息
'''
def _fmtEncode(fmt):
    '''Encode format code.'''
    pass

'''
编码数据
'''
def _encode(data):
    '''
    Encode the input data stream.
    Add mode prefix, encode data using ISO-8859-1,
    group data, add padding suffix, and call RS encoding method.
    '''
    if len(data) > 17:
        raise CapacityOverflowException(
                'Error: Version 1 QR code encodes no more than 17 characters.')

'''
将 数据填充到模板中
'''
def _fillData(bitstream):
    '''Fill the encoded data into the template QR code matrix'''
    pass

'''
应用掩码
'''
def _mask(mat):
    '''
    Mask the data QR code matrix with all 8 masks,
    and select the best mask.
    '''
    pass

'''
填充格式信息
'''
def _fillInfo(arg):
    '''
    Fill the encoded format code into the masked QR code matrix.
    '''
    pass

'''
创建最终的QR码矩阵
'''
def _genBitmap(bitstream):
    '''
    Take in the encoded data stream and generate the
    final QR code bitmap.
    '''
    return _fillInfo(_mask(_fillData(bitstream)))

print "------------ end, %r ------------" % aa.count('b')

test = [[ (i+j)%2 for i in range(8) ] for j in range(8)]

_genImage(test, 240, 'test.png')

