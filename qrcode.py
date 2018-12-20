#!/usr/bin/python
#coding:utf-8

from PIL import Image, ImageDraw
from numpy import *;

import logger

#logger.ENABLE_DEBUG = False

aa = 'abbdbdbbf'

print "----------- start, %r -----------" % aa.count('b')

def _gfpMul(x, y, prim=0x11d, field_charac_full=256, carryless=True):
    '''Galois field GF(2^8) multiplication.'''
    r = 0
    while y:
        if y & 1:
            r = r ^ x if carryless else r + x
        y = y >> 1
        x = x << 1
        if prim > 0 and x & field_charac_full:
            x = x ^ prim
    return r

# Calculate alphas to simplify GF calculations.
_gfExp = [0] * 512
_gfLog = [0] * 256
_gfPrim = 0x11d

_x = 1

for i in range(255):
    _gfExp[i] = _x
    _gfLog[_x] = i
    _x = _gfpMul(_x, 2)

for i in range(255, 512):
    _gfExp[i] = _gfExp[i-255]

def _gfPow(x, pow):
    '''GF power.'''
    return _gfExp[(_gfLog[x] * pow) % 255]

def _gfMul(x, y):
    '''Simplified GF multiplication.'''
    if x == 0 or y == 0:
        return 0
    return _gfExp[_gfLog[x] + _gfLog[y]]

def _gfPolyMul(p, q):
    '''GF polynomial multiplication.'''
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i+j] ^= _gfMul(p[i], q[j])
    return r

def _gfPolyDiv(dividend, divisor):
    '''GF polynomial division.'''
    res = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = res[i]
        if coef != 0:
            for j in range(1, len(divisor)):
                if divisor[j] != 0:
                    res[i+j] ^= _gfMul(divisor[j], coef)
    sep = -(len(divisor) - 1)
    return res[:sep], res[sep:]

def _rsGenPoly(nsym):
    '''Generate generator polynomial for RS algorithm.'''
    g = [1]
    for i in range(nsym):
        g = _gfPolyMul(g, [1, _gfPow(2, i)])
    return g

def _rsEncode(bitstring, nsym):
    '''Encode bitstring with nsym EC bits using RS algorithm.'''
    gen = _rsGenPoly(nsym)
    res = [0] * (len(bitstring) + len(gen) - 1)
    res[:len(bitstring)] = bitstring
    for i in range(len(bitstring)):
        coef = res[i]
        if coef != 0:
            for j in range(1, len(gen)):
                res[i+j] ^= _gfMul(gen[j], coef)
    res[:len(bitstring)] = bitstring
    return res

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
    logger.dbg()

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
    logger.dbg("bitmap=%r", len(bitmap))
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

    #
    # 1. 添加 模式指示符;
    # Byte mode prefix 0100.
    #
    bitstring = '0100'

    #
    # 2. 添加 字符数指示符;
    # Character count in 8 binary bits.
    #
    bitstring += '{:08b}'.format(len(data))

    #
    # 3. 把每一个字符 用 ISO/IEC 8859-1 标准编码, 然后 转换为 八位的二进制;
    # Encode every character in ISO-8859-1 in 8 binary bits.
    #
    for c in data:
        bitstring += '{:08b}'.format(ord(c.encode('iso-8859-1')))

    #
    # 4. 添加终止符
    # Terminator 0000.
    #
    bitstring += '0000'

    res = list()
    #
    # 5. 把 每8位 二进制数据 转换为 整数;
    # Convert string to byte numbers.
    #
    while bitstring:
        res.append(int(bitstring[:8], 2))
        bitstring = bitstring[8:]

    #
    # 6. 如果编码后的数据不足版本及纠错级别的最大容量, 则在尾部补充 "11101100" 和 "00010001"
    # Add padding pattern.
    #
    while len(res) < 19:
        res.append(int('11101100', 2))
        res.append(int('00010001', 2))

    #
    # 7. 截取 前19个字符
    # Slice to 19 bytes for V1-L.
    #
    res = res[:19]

    #
    # 8. 添加 7 bit EC;
    # Call _rsEncode to add 7 EC bits.
    #
    return _rsEncode(res, 7)

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


