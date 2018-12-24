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

def _matCp(src, dst, y_coordinate, x_coordinate):
    '''
    把 矩阵src 拷贝到 矩阵dst里, 起始点(x, y)

    --------> X
    |
    |
    |
    v
    Y

    Copy the content of matrix src into matrix dst.
    The top-left corner of src is positioned at (x, y)
    in dst.
    '''

    #logger.dbg("x=%r, y=%r", x_coordinate, y_coordinate)
    #logger.dbg("src-len=%r\ndata=%r", len(src), src)
    #logger.dbg("dst-len=%r\ndata=%r", len(dst), dst)

    res = copy.deepcopy(dst)

    for j in range(len(src)):
        for i in range(len(src[0])):
            #logger.dbg("j=%r, i=%r\n", j, i)
            res[y_coordinate+j][x_coordinate+i] = src[j][i]

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

'''
创建 代表数据区域的"蒙版"
    避免 掩码图案 把 功能性区域 给 过滤掉;

# Data area mask to avoid applying masks to functional area.
'''
_dataAreaMask = [[_DARK for i in range(21)] for j in range(21)]
#左上角: 9x9 的 白色;
_dataAreaMask = _matCp([[_LIGHT for i in range(9)] for j in range(9)],
    _dataAreaMask, 0, 0)
#左下角: 9x8 的 白色;
_dataAreaMask = _matCp([[_LIGHT for i in range(9)] for j in range(8)],
    _dataAreaMask, 13, 0)
#右上角: 8x9 的 白色;
_dataAreaMask = _matCp([[_LIGHT for i in range(8)] for j in range(9)],
    _dataAreaMask, 0, 13)
#水平的 4个 定位图形
_dataAreaMask = _matCp([[_LIGHT for i in range(4)]], _dataAreaMask, 6, 9)
#垂直的 4个 定位图形
_dataAreaMask = _matCp([[_LIGHT] for i in range(4)], _dataAreaMask, 9, 6)
#logger.dbg("dataArea-Mask-len=%r, data=%r", len(_dataAreaMask), _dataAreaMask)

def _matAnd(mat1, mat2):
    '''
    使用 '蒙板 mat1' 和 '掩码 mat2' 进行 或运算, 
        因为 白色部分是 1;

    Matrix-wise and.
    Dark and dark -> dark
    Light and light -> light
    Dark and light -> light
    Light and dark -> light
    '''
    #logger.dbg("data-mat1, len=%r, data=%r", len(mat1), mat1)
    #logger.dbg("data-mat1-0, len=%r, data=%r", len(mat1[0]), mat1[0])

    res = [[_LIGHT for i in range(len(mat1[0]))] for j in range(len(mat1))]

    for j in range(len(mat1)):
        for i in range(len(mat1[0])):
            res[j][i] = int(mat1[j][i] == _LIGHT or mat2[j][i] == _LIGHT)

    #logger.dbg("data, len=%r, data=%r", len(res), res)
    return res

'''
定义 掩码, 和 蒙板 进行 '与' 动作,
    避免 掩码图案 把 功能性区域 给 过滤掉;

# Data masks defined in QR standard.
'''
_dataMasks = []

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if (i+j)%2==0 else _LIGHT for i in range(21)] for j in range(21)]))
dataMask_000 =    _matAnd(_dataAreaMask,
    [[_DARK if (i+j)%2==0 else _LIGHT for i in range(21)] for j in range(21)])
#logger.dbg("data-Mask-len=%r, data=%r", len(dataMask_000), dataMask_000)

_dataMasks.append(_matAnd(_dataAreaMask,
    [[_DARK if j%2==0 else _LIGHT for i in range(21)] for j in range(21)]))
dataMask_001 =    _matAnd(_dataAreaMask,
    [[_DARK if j%2==0 else _LIGHT for i in range(21)] for j in range(21)])

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
dataMask_007 =    _matAnd(_dataAreaMask,
    [[_DARK if ((i+j)%2+(i*j)%3)%2==0 else _LIGHT for i in range(21)] for j in range(21)])
#logger.dbg("all Mask-type, len=%r, data=%r", len(_dataMasks), _dataMasks)

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
    '''
    添加 里德-所罗门码(Reed-Solomon)
        里德-所罗门码是定长码:
        也就是一个固定长度输入的数据, 将被处理成一个固定长度的输出数据,
        在最常用的(255,223)里所码中, 223个里德-所罗门输入符号(每个符号有8个位元)被编码成255个输出符号;

    Encode bitstring with nsym EC bits using RS algorithm.
    '''
    gen = _rsGenPoly(nsym)
    res = [0] * (len(bitstring) + len(gen) - 1)
    res[:len(bitstring)] = bitstring
    for i in range(len(bitstring)):
        coef = res[i]
        if coef != 0:
            for j in range(1, len(gen)):
                res[i+j] ^= _gfMul(gen[j], coef)
    res[:len(bitstring)] = bitstring

    logger.dbg("encode+rscode, len:%r, data=\n\t%r\n", len(res), res)

    return res

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
    #logger.dbg("block:%rx%r", len(bitmap), len(bitmap))

    #用图像宽度 除以 矩阵维度得到 QR码中一个单位对应的像素数
    a_unit_size = qrcodesize / len(bitmap)
    #logger.dbg("a-rectangle-size=%r", a_unit_size)

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

def _penalty(mat):
    '''
    Calculate penalty score for a masked matrix.
    N1: penalty for more than 5 consecutive pixels in row/column,
        3 points for each occurrence of such pattern,
        and extra 1 point for each pixel exceeding 5
        consecutive pixels.
    N2: penalty for blocks of pixels larger than 2x2.
        3*(m-1)*(n-1) points for each block of mxn
        (larger than 2x2).
    N3: penalty for patterns similar to the finder pattern.
        40 points for each occurrence of 1:1:3:1:1 ratio
        (dark:light:dark:light:dark) pattern in row/column,
        preceded of followed by 4 consecutive light pixels.
    N4: penalty for unbalanced dark/light ratio.
        10*k points where k is the rating of the deviation of
        the proportion of dark pixels from 50% in steps of 5%.
    '''
    # Initialize.
    n1 = n2 = n3 = n4 = 0
    # Calculate N1.
    for j in range(len(mat)):
        count = 1
        adj = False
        for i in range(1, len(mat)):
            if mat[j][i] == mat[j][i-1]:
                count += 1
            else:
                count = 1
                adj = False
            if count >= 5:
                if not adj:
                    adj = True
                    n1 += 3
                else:
                    n1 += 1
    for i in range(len(mat)):
        count = 1
        adj = False
        for j in range(1, len(mat)):
            if mat[j][i] == mat[j-1][i]:
                count += 1
            else:
                count = 1
                adj = False
            if count >= 5:
                if not adj:
                    adj = True
                    n1 += 3
                else:
                    n1 += 1
    # Calculate N2.
    m = n = 1
    for j in range(1, len(mat)):
        for i in range(1, len(mat)):
            if mat[j][i] == mat[j-1][i] and mat[j][i] == mat[j][i-1] and mat[j][i] == mat[j-1][i-1]:
                if mat[j][i] == mat[j-1][i]:
                    m += 1
                if mat[j][i] == mat[j][i-1]:
                    n += 1
            else:
                n2 += 3 * (m-1) * (n-1)
                m = n = 1
    # Calculate N3.
    count = 0
    for row in mat:
        rowstr = ''.join(str(e) for e in row)
        occurrences = []
        begin = 0
        while rowstr.find('0100010', begin) != -1:
            begin = rowstr.find('0100010', begin) + 7
            occurrences.append(begin)
        for begin in occurrences:
            if rowstr.count('00000100010', begin-4) != 0 or rowstr.count('01000100000', begin) != 0:
                count += 1
    transposedMat = _transpose(mat)
    for row in transposedMat:
        rowstr = ''.join(str(e) for e in row)
        occurrences = []
        begin = 0
        while rowstr.find('0100010', begin) != -1:
            begin = rowstr.find('0100010', begin) + 7
            occurrences.append(begin)
        for begin in occurrences:
            if rowstr.count('00000100010', begin-4) != 0 or rowstr.count('01000100000', begin) != 0:
                count += 1
    n3 += 40 * count
    # Calculate N4.
    dark = sum(row.count(_DARK) for row in mat)
    percent = int((float(dark) / float(len(mat)**2)) * 100)
    pre = percent - percent % 5
    nex = percent + 5 - percent % 5
    n4 = min(abs(pre-50)/5, abs(nex-50)/5) * 10

    return n1 + n2 + n3 + n4

def _mask(mat):
    '''
    应用掩码, 返回 结果矩阵 和 掩码ID;

    Mask the data QR code matrix with all 8 masks,
    call _penalty to calculate penalty scores for each
    and select the best mask.
    Return tuple(selected masked matrix, number of selected mask).
    '''
    #logger.dbg("data-len=%r, data=%r", len(mat), mat)

    maskeds = [_matXor(mat, dataMask) for dataMask in _dataMasks]
    #logger.dbg("Mask-len=%r, data=%r", len(maskeds), maskeds)
    penalty = [0] * 8

    for i, masked in enumerate(maskeds):
        penalty[i] = _penalty(masked)

    logger.dbg("penalty-len=%r, data=%r", len(penalty), penalty)

    # Find the id of the best mask.
    index = penalty.index(min(penalty))

    return maskeds[index], index

def _fmtEncode(fmt):
    '''
    获得具有EC位的15位 [格式信息],
        实现容错码计算 和 应用掩码;

    Encode the 15-bit format code using BCH code.
    '''
    logger.dbg("format-info=%r", fmt)

    g = 0x537
    code = fmt << 10
    for i in range(4, -1, -1):
        if code & (1 << (i+10)):
            code ^= g << i
    #
    # 计算得出十位BCH容错码接在格式信息之后,
    #   还要与掩码101010000010010进行异或, 作用同QR掩码;
    #
    return ((fmt << 10) ^ code) ^ 0b101010000010010

def _fillInfo(arg):
    '''
    填充格式信息

    Fill the encoded format code into the masked QR code matrix.
    arg: (masked QR code matrix, mask number).
    '''
    mat, mask = arg

    logger.dbg("mask-id=%r", mask)

    #
    # 1. 计算格式信息
    #

    # 01 is the format code for L error control level,
    # concatenated with mask id and passed into _fmtEncode
    # to get the 15 bits format code with EC bits.
    fmt = _fmtEncode(int('01'+'{:03b}'.format(mask), 2))
    logger.dbg("fmt-len=%r", fmt)

    #
    # 2. 把 格式信息取反, 然后转换为 15bit的二进制
    #
    fmtarr = [[not int(c)] for c in '{:015b}'.format(fmt)]
    logger.dbg("revert-fmt-len=%r, data=%r", len(fmtarr), fmtarr)

    # 从后 往前:

    # 3.1. 填充 水平的 0~7
    horizontal_0_7 = fmtarr[7:]
    #logger.dbg("01, fmt-len=%r, data=%r", len(horizontal_0_7), horizontal_0_7)
    mat = _matCp(_transpose(horizontal_0_7), mat, 8, 13)

    # 3.2. 填充 垂直的 0~5
    vertical_0_5 = fmtarr[9:][::-1]
    mat = _matCp(vertical_0_5, mat, 0, 8)
    #logger.dbg("02, fmt-len=%r, data=%r", len(vertical_0_5), vertical_0_5)

    # 3.3. 填充 垂直的 6~7
    mat = _matCp(fmtarr[7:9][::-1], mat, 7, 8)

    # 3.4. 填充 垂直的 8~14
    mat = _matCp(fmtarr[:7][::-1], mat, 14, 8)

    # 3.5. 填充 水平的 9~14
    mat = _matCp(_transpose(fmtarr[:6]), mat, 8, 0)

    # 3.6. 填充 水平的 8
    mat = _matCp([fmtarr[6]], mat, 8, 7)

    return mat

def _genBitmap(bitstream):
    '''
    创建最终的QR码矩阵

    Take in the encoded data stream and generate the
    final QR code bitmap.
    '''
    return _fillInfo(_mask(_fillData(bitstream)))

class CapacityOverflowException(Exception):
    '''Exception for data larger than 17 characters in V1-L byte mode.'''
    def __init__(self, arg):
        self.arg = arg

    def __str__(self):
        return repr(self.arg)

def _encode(data):
    '''
    编码 输入数据,
        返回 一维 整数矩阵 [ 模式指示符 + 字数指示符 + 数据内容 + 终止符 + 容错码 ]

    Encode the input data stream.
    Add mode prefix, encode data using ISO-8859-1,
    group data, add padding suffix, and call RS encoding method.
    '''
    logger.dbg("input-data, len=%r, [%r]", len(data), data)

    #
    # 检测输入的数据是否超过V1-L byte mode 的最大编码长度17,
    #       如果超过就抛出异常
    #
    if len(data) > 17:
        raise CapacityOverflowException(
                'Error: Version 1 QR code[binary mode] encodes no more than 17 characters.')
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
    #       ISO-8859-1编码是单字节编码，向下兼容ASCII;
    #
    # Encode every character in ISO-8859-1 in 8 binary bits.
    #
    for c in data:
        bitstring += '{:08b}'.format(ord(c.encode('iso-8859-1')))
    logger.dbg("byte mode + char cnt + data=\n\t%r\n", bitstring)

    #
    # 4. 添加终止符,
    #       如果尾部数据不足8bit，则在尾部 填充0
    #
    # Terminator 0000.
    #
    tmpstr = bitstring
    convert_str_to_8_bit_array = ''
    last_str = ''
    while tmpstr:
        convert_str_to_8_bit_array += tmpstr[:8]
        last_str = tmpstr[:8]
        convert_str_to_8_bit_array += ', '
        tmpstr = tmpstr[8:]
    #logger.dbg("8 bit to arry=\n\t%r\n", convert_str_to_8_bit_array)
    logger.dbg("last-str, len=%r, data=%r, append-0=%r", len(last_str), last_str, last_str+'0'*(8-len(last_str)))

    bitstring += '0'*(8-len(last_str))
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
    # 6. 如果编码后的数据不足版本及纠错级别的最大容量,
    #       则在尾部补充 "11101100" 和 "00010001"
    #
    # Add padding pattern.
    #
    while len(res) < 19: #zgj, 这个19是如何计算的;
        res.append(int('11101100', 2))
        res.append(int('00010001', 2))

    #
    # 7. 截取 前19个字符
    # Slice to 19 bytes for V1-L.
    #
    res = res[:19]

    logger.dbg("value:1~19, data=\n\t%r\n", res)

    #
    # 8. 添加 RS容错码;
    #
    return _rsEncode(res, 7) #zgj, 为什么是7个;

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

def _matXor(mat1, mat2):
    '''
    数据 添加 掩码

    Matrix-wise xor.
    Dark xor dark -> light
    Light xor light -> light
    Dark xor light -> dark
    Light xor dark -> dark
    '''
    res = [[_LIGHT for i in range(len(mat1[0]))] for j in range(len(mat1))]

    for j in range(len(mat1)):
        for i in range(len(mat1[0])):
            res[j][i] = int(mat1[j][i] == mat2[j][i])

    return res

def qrcode(data, width=210, filename='QR-code.png'):
    '''
    创建 二维码

    Module public interface
    '''

    try:
        _genImage(_genBitmap(_encode(data)), width, filename)
    except Exception, e:
        print e
        raise e

print "------------ end ------------"

################### main #####################
'''
#001-test
test = [[ (i+j)%2 for i in range(8) ] for j in range(8)]
_genImage(test, 240, 'test.png')

#002-test
data = 'test'
filledMat = _fillData(_encode(data))
_genImage(filledMat, 210, "test01.png")

#003-test
mask = [[ _DARK if (i+j)%2 == 0 else _LIGHT for i in range(21) ] for j in range(21)]
_genImage(mask, 210, "test01.png")

#004-test
_genImage(_dataAreaMask, 210, "filter_mask.png")

#005-test
_genImage(dataMask_000, 210, "test_000_mask.png")

#006-test
_genImage(dataMask_001, 210, "test_001_mask.png")

#007-test
_genImage(dataMask_007, 210, "test_007_mask.png")

dullData = '00000000000000000'

filledMat = _fillData(_encode(dullData))
_genImage(filledMat, 210, "test01.png")

filledMat = _fillData(_encode(dullData))
maskedMat, maskID = _mask(filledMat)
_genImage(maskedMat, 210, "test02.png")
logger.dbg("mask-id=%r", maskID)
'''

qrcode('11115111101111567')

