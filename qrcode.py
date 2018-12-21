#!/usr/bin/python
#coding:utf-8

from PIL import Image, ImageDraw
#from numpy import *;
import copy
import sys

import logger

logger.ENABLE_DEBUG = True
#logger.PRINT_TAG = True
#logger.PRINT_BUILDTIME = True
#logger.PRINT_FILENAME = True



print "------------ start ------------"

'''
0:对应黑色, 1:对应白色
'''
_DARK = 0
_LIGHT = 1


def _transpose(mat):
    '''
    转换矩阵, 例如: 1x5 --> 5x1

    Transpose a matrix
    '''
    res = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
    return res

def _timSeq(time_len, vertical=False):
    '''
    创建 定位图形 矩阵

    Generate a horizontal, unless specified vertical
    timing sequence with alternating dark and light
    pixels with length time_len.
    '''
    res = [[i % 2 for i in range(time_len)]]
    #logger.dbg("time-len=%d, data=%r", len(res), res)

    if vertical:
        res = _transpose(res)
        #logger.dbg("vertical-time-len=%d, data=%r", len(res), res)

    return res

def _matCp(src, dst, top, left):
    '''
    把 矩阵src 拷贝到 矩阵dst里, 起始点(left, top)

    Copy the content of matrix src into matrix dst.
    The top-left corner of src is positioned at (left, top)
    in dst.
    '''

    #logger.dbg("top=%r, left=%r", top, left)
    #logger.dbg("src-len=%r\ndata=%r", len(src), src)
    #logger.dbg("dst-len=%r\ndata=%r", len(dst), dst)

    res = copy.deepcopy(dst)

    for j in range(len(src)):
        for i in range(len(src[0])):
            #logger.dbg("j=%r, i=%r\n", j, i)
            res[top+j][left+i] = src[j][i]

    #logger.dbg("last-len=%r\n\tdata=%r\n", len(res), res)
    return res

'''
创建 位置探测图形 矩阵

Finder pattern.
'''
_finder = _matCp(
        _matCp(
            [[_DARK for i in range(3)] for j in range(3)],
            [[_LIGHT for i in range(5)] for j in range(5)],
            1,
            1),
        [[_DARK for i in range(7)] for j in range(7)],
        1,
        1)

'''
创建 校正图形 矩阵

Alignment pattern. Not used in version 1.
'''
_align = _matCp(
        _matCp([[_DARK]],
            [[_LIGHT for i in range(3)] for j in range(3)],
            1,
            1),
        [[_DARK for i in range(5)] for j in range(5)],
        1,
        1)

'''
初始化一个 21x21的 白色矩阵
'''
# Version 1 QR code template with fixed patterns.
_ver1 = [[_LIGHT for i in range(21)] for j in range(21)]
#logger.dbg("verl-len=%r", len(_ver1))
#logger.dbg("finder-len=%r", len(_finder))

# 添加 左上角
_ver1 = _matCp(_finder, _ver1, 0, 0)
# 添加 右上角
_ver1 = _matCp(_finder, _ver1, len(_ver1)-len(_finder), 0)
# 添加 左下角
_ver1 = _matCp(_finder, _ver1, 0, len(_ver1)-len(_finder))

# 添加 水平 定位图形
_ver1 = _matCp(_timSeq(len(_ver1)-len(_finder)-len(_finder)-2), _ver1, 6, 8)
# 添加 垂直 定位图形
_ver1 = _matCp(_timSeq(len(_ver1)-len(_finder)-len(_finder)-2, vertical=True), _ver1, 8, 6)

# 添加 固定的 一个黑点
_ver1 = _matCp([[_DARK]], _ver1, len(_ver1)-len(_finder)-1, len(_finder)+1)

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

    logger.dbg("data-len:%d, data=\n\t%r\n", len(res), res)

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
    logger.dbg("block:%rx%r", len(bitmap), len(bitmap))

    #用图像宽度 除以 矩阵维度得到 QR码中一个单位对应的像素数
    a_unit_size = qrcodesize / len(bitmap)
    logger.dbg("a-rectangle-size=%r", a_unit_size)

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

def _encode(data):
    '''
    编码数据, 返回 一维 整数矩阵

    Encode the input data stream.
    Add mode prefix, encode data using ISO-8859-1,
    group data, add padding suffix, and call RS encoding method.
    '''
    logger.dbg()

    if len(data) > 17:
        raise CapacityOverflowException(
                'Error: Version 1 QR code encodes no more than 17 characters.')

    #
    # 1. 添加 模式指示符;
    # Byte mode prefix 0100.
    #
    bitstring = '0100'
    logger.dbg("byte mode=\n\t%r\n", bitstring)

    #
    # 2. 添加 字符数指示符;
    # Character count in 8 binary bits.
    #
    bitstring += '{:08b}'.format(len(data))
    logger.dbg("byte mode + char cnt=\n\t%r\n", bitstring)

    #
    # 3. 把每一个字符 用 ISO/IEC 8859-1 标准编码, 然后 转换为 八位的二进制;
    # Encode every character in ISO-8859-1 in 8 binary bits.
    #
    for c in data:
        bitstring += '{:08b}'.format(ord(c.encode('iso-8859-1')))
    logger.dbg("byte mode + char cnt + data=\n\t%r\n", bitstring)

    #
    # 4. 添加终止符
    # Terminator 0000.
    #
    bitstring += '0000'
    logger.dbg("byte mode + char cnt + data + terminater=\n\t%r\n", bitstring)

    res = list()
    #
    # 5. 把 每8位 二进制数据 转换为 整数;
    # Convert string to byte numbers.
    #
    while bitstring:
        res.append(int(bitstring[:8], 2))
        bitstring = bitstring[8:]
    logger.dbg("convert byte to int=\n\t%r\n", res)

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

    logger.dbg("value:1~19, data=\n\t%r\n", res)

    #
    # 8. 添加 7 个 EC;
    # Call _rsEncode to add 7 EC bits.
    #
    return _rsEncode(res, 7)

def _fillByte(byte, downwards=False):
    '''
     Upwards         Downwards
    ---------        ---------   
    | 0 | 1 |        | 6 | 7 |    
    ---------        ---------   
    | 2 | 3 |        | 4 | 5 |    
    ---------        ---------   
    | 4 | 5 |        | 2 | 3 |    
    ---------        ---------   
    | 6 | 7 |        | 0 | 1 |    
    ---------        ---------   

    wanghai
    ---------        ---------   
    | 7 | 6 |        | 1 | 0 |    
    ---------        ---------   
    | 5 | 4 |        | 3 | 2 |    
    ---------        ---------   
    | 3 | 2 |        | 5 | 4 |    
    ---------        ---------   
    | 1 | 0 |        | 7 | 6 |    
    ---------        ---------   

    Fill a byte into a 2 by 4 matrix upwards,
    unless specified downwards.
    '''
    bytestr = '{:08b}'.format(byte)
    res = [[0, 0], [0, 0], [0, 0], [0, 0]]

    for i in range(8):
        res[i/2][i%2] = not int(bytestr[7-i])

    #logger.dbg("a-byte-len=%d, data=%r", len(res), res)
    if downwards:
        res = res[::-1]
        #logger.dbg("revers:len=%d, data=%r", len(res), res)

    #print "\n"
    return res

def _fillData(bitstream):
    '''
    将 数据填充到模板中
        V1-L: 一共有26个数据

    Fill the encoded data into the template QR code matrix
    '''
    res = copy.deepcopy(_ver1)

    #
    # 1. 填充 第一部分数据, 一共15个数据, 0~14
    #    每一个数据是8个bit, 每一个bit占用 一个块;
    #
    for i in range(15):
        res = _matCp(_fillByte(bitstream[i], (i/3)%2!=0),
            res,
            21-4*((i%3-1)*(-1)**((i/3)%2)+2), #先向上, 后向下
            21-2*(i/3+1)) #向左 移动2个单位:块

    # 2. 填充 第二部分数据, 1个, 被 水平定位图像 分割 为 两部分;
    tmp = _fillByte(bitstream[15])
    res = _matCp(tmp[2:], res, 7, 11)
    res = _matCp(tmp[:2], res, 4, 11)

    #logger.dbg("15, len=%d, data=%r", len(tmp), tmp)

    # 3. 填充 第三部分数据, 1个;
    tmp = _fillByte(bitstream[16])
    res = _matCp(tmp, res, 0, 11)

    # 4. 填充 第四部分数据, 1个;
    tmp = _fillByte(bitstream[17], True)
    res = _matCp(tmp, res, 0, 9)

    # 5. 填充 第五部分数据, 1个, 被 水平定位图像 分割 为 两部分;
    tmp = _fillByte(bitstream[18], True)
    res = _matCp(tmp[:2], res, 4, 9)
    res = _matCp(tmp[2:], res, 7, 9)

    # 6. 填充 第六部分数据, 3个;
    for i in range(3):
        res = _matCp(_fillByte(bitstream[19+i], True),
            res, 9+4*i, 9)

    # 7. 填充 第七部分数据, 1个;
    tmp = _fillByte(bitstream[22])
    res = _matCp(tmp, res, 9, 7)

    # 8. 填充 第八部分数据, 最后 3个;
    for i in range(3):
        res = _matCp(_fillByte(bitstream[23+i], i%2==0),
            res, 9, 4-2*i)

    '''
    #sys.exit()
    '''
    return res

def _matAnd(mat1, mat2):
    '''
    Matrix-wise and.
    Dark and dark -> dark
    Light and light -> light
    Dark and light -> light
    Light and dark -> light
    '''
    res = [[_LIGHT for i in range(len(mat1[0]))] for j in range(len(mat1))]
    for j in range(len(mat1)):
        for i in range(len(mat1[0])):
            res[j][i] = int(mat1[j][i] == _LIGHT or mat2[j][i] == _LIGHT)
    return res

# Data area mask to avoid applying masks to functional area.
_dataAreaMask = [[_DARK for i in range(21)] for j in range(21)]
_dataAreaMask = _matCp([[_LIGHT for i in range(9)] for j in range(9)],
    _dataAreaMask, 0, 0)
_dataAreaMask = _matCp([[_LIGHT for i in range(9)] for j in range(8)],
    _dataAreaMask, 13, 0)
_dataAreaMask = _matCp([[_LIGHT for i in range(8)] for j in range(9)],
    _dataAreaMask, 0, 13)
_dataAreaMask = _matCp([[_LIGHT for i in range(4)]], _dataAreaMask, 6, 9)
_dataAreaMask = _matCp([[_LIGHT] for i in range(4)], _dataAreaMask, 9, 6)
logger.dbg("dataArea-Mask-len=%r, data=%r", len(_dataAreaMask), _dataAreaMask)

# Data masks defined in QR standard.
_dataMasks = []

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if (i+j)%2==0 else _LIGHT for i in range(21)] for j in range(21)]))

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if j%2==0 else _LIGHT for i in range(21)] for j in range(21)]))

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if i%3==0 else _LIGHT for i in range(21)] for j in range(21)]))
_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if (i+j)%3==0 else _LIGHT for i in range(21)] for j in range(21)]))
_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if (j/2 + i/3)%2==0 else _LIGHT for i in range(21)] for j in range(21)]))
_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if (i*j)%2+(i*j)%3==0 else _LIGHT for i in range(21)] for j in range(21)]))
_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if ((i*j)%2+(i*j)%3)%2==0 else _LIGHT for i in range(21)] for j in range(21)]))

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if ((i+j)%2+(i*j)%3)%2==0 else _LIGHT for i in range(21)] for j in range(21)]))

dataMask_000 = _matAnd(_dataAreaMask,
    [[_DARK if (i+j)%2==0 else _LIGHT for i in range(21)] for j in range(21)])

logger.dbg("data-Mask-len=%r, data=%r", len(dataMask_000), dataMask_000)

print "------------ end ------------"

################### main #####################
'''
test = [[ (i+j)%2 for i in range(8) ] for j in range(8)]
_genImage(test, 240, 'test.png')

#data = '00000000000000000'

data = 'test'
filledMat = _fillData(_encode(data))
_genImage(filledMat, 210, "test01.png")

mask = [[ _DARK if (i+j)%2 == 0 else _LIGHT for i in range(21) ] for j in range(21)]
_genImage(mask, 210, "test01.png")

_genImage(_dataAreaMask, 210, "test01.png")

'''

_genImage(dataMask_000, 210, "test01.png")



