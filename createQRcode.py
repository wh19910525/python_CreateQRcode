#!/usr/bin/python
#coding:utf-8

from PIL import Image, ImageDraw

import copy
import sys

import logger

logger.ENABLE_DEBUG = True
#logger.PRINT_TAG = True
#logger.PRINT_BUILDTIME = True
#logger.PRINT_FILENAME = True

numeric_mode = 1
alphanumeric_mode = 2
mixed_mode = 3
chinese_mode = 4 
byte_mode = 9
#########################
curEncodeMode = mixed_mode

print "------------ start ------------"

'''
1:对应黑色, 0:对应白色
'''
DARK_IS_1 = True

if vars().has_key('DARK_IS_1') and DARK_IS_1:
    pass
else:
    DARK_IS_1 = False

if DARK_IS_1:
    _DARK = 1
    _LIGHT = 0
else:
    _DARK = 0
    _LIGHT = 1
logger.dbg("dark:%r", _DARK)

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
创建 一个 21x21的 (功能图形)矩阵
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

def _matAnd(maskMat, qrMaskMat):
    '''
    创建 二维码掩码,

        使用 '蒙板 maskMat' 和 '掩码 qrMaskMat' 进行 [与]运算,
        因为 白色部分是 1;

    Matrix-wise and.
    Dark and dark -> dark
    Light and light -> light
    Dark and light -> light
    Light and dark -> light
    '''
    #logger.dbg("data-maskMat, len=%r, data=%r", len(maskMat), maskMat)
    #logger.dbg("data-maskMat-0, len=%r, data=%r", len(maskMat[0]), maskMat[0])

    res = [[_LIGHT for i in range(len(maskMat[0]))] for j in range(len(maskMat))]

    for j in range(len(maskMat)):
        for i in range(len(maskMat[0])):
            if DARK_IS_1:
                res[j][i] = int(maskMat[j][i] and qrMaskMat[j][i])
            else:
                res[j][i] = int(maskMat[j][i] or  qrMaskMat[j][i])

    #logger.dbg("data, len=%r, data=%r", len(res), res)
    return res

'''
定义 掩码 矩阵, 8种:

    掩码 和 蒙板 进行 '与' 动作,

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
def _genImage(bitmapMat, qrcodesize, filename):
    '''
    创建 二维码

    以矩阵的左上角为原点, 原点坐标定义为(0, 0), x 轴向右，坐标 x 对应列,
        y 轴向下, 坐标 y 对应行;
        于是对于图像中的像素(x, y), 有矩阵元素 mat [ y ] [ x ] 与之对应。
    --------> X
    |
    |
    |
    v
    Y

    Generate image corresponding to the input bitmapMat
    with specified qrcodesize and filename.
    '''
    #logger.dbg("last-data-len=%r, data=%r", len(bitmapMat), bitmapMat)

    width = qrcodesize
    height = qrcodesize

    # New image in black-white mode initialized with white.
    img = Image.new('1', (width, height), 'white')
    drw = ImageDraw.Draw(img)

    # Normalized pixel width.
    logger.dbg("block:%rx%r", len(bitmapMat[0]), len(bitmapMat))

    #用图像宽度 除以 矩阵维度得到 QR码中一个单位对应的像素数
    a_unit_size = qrcodesize / len(bitmapMat)
    logger.dbg("a-rectangle-size=%r", a_unit_size)

    for y in range(height):
        # Normalized y coordinate in bitmapMat
        normaly = y / a_unit_size

        for x in range(width):
            # Normalized x coordinate in bitmapMat
            normalx = x / a_unit_size

            if normaly < len(bitmapMat) and normalx < len(bitmapMat):
                #
                # 在 ImageDraw里, 0(False)是黑色，1(True)是白色；
                #
                if DARK_IS_1:
                    drow_color = not bitmapMat[normaly][normalx]
                else:
                    drow_color = bitmapMat[normaly][normalx]

                # Draw pixel.
                drw.point((x, y), fill=drow_color)

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
    为 矩阵数据 应用掩码, 返回 结果矩阵 和 掩码ID;

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

def _createFmtEncode(fmt):
    '''
    创建完整的格式信息

    1. 首先 获得具有EC位的15位 [格式信息],

    Encode the 15-bit format code using BCH code.
    '''
    logger.dbg("version+mask=%r", '{:05b}'.format(fmt))

    g = 0x537
    code = fmt << 10
    for i in range(4, -1, -1):
        if code & (1 << (i+10)):
            code ^= g << i
    '''
    2. 把 得到 十位BCH容错码 接在格式信息之后, 一共15个数字,
        还要与掩码101010000010010进行异或, 作用同QR掩码;
    '''
    formatInfo = (fmt << 10) ^ code
    formatInfoMask = 0b101010000010010
    finalFormatInfo = formatInfo ^ formatInfoMask

    logger.dbg("format-info      =%r", '{:015b}'.format(formatInfo))
    logger.dbg("format-info-mask =%r", '{:015b}'.format(formatInfoMask))
    logger.dbg("final-format-info=%r\n", '{:015b}'.format(finalFormatInfo))

    return finalFormatInfo

def _fillFormatInfo(arg):
    '''
    填充格式信息

    Fill the encoded format code into the masked QR code matrix.
    arg: (masked QR code matrix, mask number).
    '''
    mat, mask = arg

    logger.dbg("mask-id=%r", mask)

    #
    # 1. 计算 15位 格式信息
    #

    # 01 is the format code for L error control level,
    # concatenated with mask id and passed into _createFmtEncode
    # to get the 15 bits format code with EC bits.
    fmt = _createFmtEncode(int('01'+'{:03b}'.format(mask), 2))
    #logger.dbg("fmt=%r", fmt)

    #
    # 2. 把 格式信息 转换为 15bit的二进制
    #
    # bit 0 ~ bit 4, 一共5位: bit0, bit1 为 [容错等级], bit2, bit3, bit4 为 [掩码类型];
    #
    if DARK_IS_1:
        fmtarr = [[int(c)] for c in '{:015b}'.format(fmt)]
    else:
        fmtarr = [[not int(c)] for c in '{:015b}'.format(fmt)]
    logger.dbg("fmtInfo-len=%r, data=%r", len(fmtarr), fmtarr)

    #
    # 二维码 格式信息, 水平方向 从左向右 一共15个数字, 如下:
    #
    # ---------------------------------------------------------------------------------------------------------------------------------
    # |  14  |  13  |  12  |  11  |  10  |  9   | 空格 |   8  | 空格n个 |   7  |   6  |   5   |   4   |   3   |   2   |   1   |   0   |
    # ---------------------------------------------------------------------------------------------------------------------------------
    # | bit0 | bit1 | bit2 | bit3 | bit4 | bit5 |      | bit6 |         | bit7 | bit8 |  bit9 | bit10 | bit11 | bit12 | bit13 | bit14 |
    # ---------------------------------------------------------------------------------------------------------------------------------
    #
    # 3.1. 填充 水平的 0~7, 共 8个数字:
    horizontal_0_7 = _transpose(fmtarr[7:]) #截取 bit7 ~ bit14
    #logger.dbg("01, fmt-len=%r, data=%r", len(horizontal_0_7), horizontal_0_7)
    mat = _matCp(horizontal_0_7, mat, 8, 13)

    # 3.2. 填充 水平的 8, 共 1个数字:
    mat = _matCp([fmtarr[6]], mat, 8, 7) #获取bit6

    # 3.3. 填充 水平的 9~14, 共 6个数字:
    mat = _matCp(_transpose(fmtarr[:6]), mat, 8, 0) #截取 bit0 ~ bit5

    #
    # 格式信息, 垂直方向 从下向上 一共15个数字, 如下:
    #   14, 13, 12, 11, 10, 9, 8, 空格..., 7, 6, 空格, 5, 4, 3, 2, 1, 0
    #
    # 3.4. 填充 垂直的 0~5, 共 6个数字:
    vertical_0_5 = fmtarr[9:][::-1] #截取 bit9 ~ bit14, 然后反向
    #logger.dbg("02, fmt-len=%r, data=%r", len(vertical_0_5), vertical_0_5)
    mat = _matCp(vertical_0_5, mat, 0, 8)

    # 3.5. 填充 垂直的 6~7, 共 2个数字:
    mat = _matCp(fmtarr[7:9][::-1], mat, 7, 8) #截取 bit7 ~ bit9, 然后反向

    # 3.6. 填充 垂直的 8~14, 共 7个数字:
    mat = _matCp(fmtarr[:7][::-1], mat, 14, 8) #截取 bit0 ~ bit6, 然后反向

    return mat

def _genBitmap(bitstream):
    '''
    创建最终的QR码矩阵

    Take in the encoded data stream and generate the
    final QR code bitmap.
    '''
    return _fillFormatInfo(_mask(_fillData(bitstream)))

class CapacityOverflowException(Exception):
    '''Exception for data larger than 17 characters in V1-L byte mode.'''
    def __init__(self, arg):
        self.arg = arg

    def __str__(self):
        return repr(self.arg)

def convert_every8bit_str(binary_str):
    every8bit_str = ''

    while binary_str:
        tmp_str = binary_str[:8]
        if len(tmp_str) > 4:
            tmp_str = tmp_str[:4] + '-' + tmp_str[4:]
        every8bit_str += tmp_str

        if len(binary_str) >= 8:
            every8bit_str += ', '

        binary_str = binary_str[8:]

    #logger.dbg("8 bit to arry=\n\t%r\n", every8bit_str)

    return every8bit_str

def _encode_byte_mode(data):
    #
    # 1. 添加 模式指示符;
    # Byte mode: 0100.
    #
    bitstring = '0100'
    logger.dbg("Mode=\n\t%r\n", bitstring)

    #
    # 2. 添加 字符数指示符;
    # Character count: 8 bits.
    #
    bitstring += '{:08b}'.format(len(data))
    logger.dbg("Mode + char CNT=\n\t%r\n", convert_every8bit_str(bitstring))

    #
    # 3. 把每一个字符 用 ISO/IEC 8859-1 标准编码, 然后 转换为 八位的二进制;
    #       ISO-8859-1编码是单字节编码，向下兼容ASCII;
    #
    for c in data:
        bitstring += '{:08b}'.format(ord(c.encode('iso-8859-1')))

    logger.dbg("byte mode + char cnt + data=\n\t%r\n", convert_every8bit_str(bitstring))

    return bitstring

def _calAlphanumericMode_value(alphanumericMode_value):
    if ord(alphanumericMode_value) >= ord('0') and ord(alphanumericMode_value) <= ord('9'): # 数字
        value = ord(alphanumericMode_value) - 48
    elif ord(alphanumericMode_value) >= ord('A') and ord(alphanumericMode_value) <= ord('Z'): #大写字母
        value = ord(alphanumericMode_value) - 65 + 10

    logger.dbg("convert alphanumeric mode bit to value=%r", value)

    return value

def _encode_alphanumeric_mode(data):
    logger.dbg("\n\tdata=%r\n", data)

    #
    # 1. 添加 模式指示符;
    # Byte mode: 0010.
    #
    bitstring = '0010'
    logger.dbg("Mode=\n\t%r\n", bitstring)

    #
    # 2. 添加 字符数指示符;
    # Character count: 9 bits.
    #
    bitstring += '{:09b}'.format(len(data))
    logger.dbg("Mode + char CNT=\n\t%r\n", convert_every8bit_str(bitstring))

    #
    # 3. 对于每一个输入的字符, 它的值为 0到44;
    #
    while data:
        tmpstr = data[:2]

        if len(tmpstr) == 2:
            has2char_bit0 = tmpstr[0]
            has2char_bit1 = tmpstr[1]

            bit0_value = _calAlphanumericMode_value(has2char_bit0)
            bit1_value = _calAlphanumericMode_value(has2char_bit1)

            has2char_value = bit0_value*45 + bit1_value

            logger.dbg("has2char-value=%r", has2char_value)

            converting_every_integer_to_binary = '{:011b}'.format(has2char_value)
        elif len(tmpstr) == 1:
            has_onlychar_bit0 = tmpstr[0]

            bit0_value = _calAlphanumericMode_value(has_onlychar_bit0)

            logger.dbg("onlychar-value=%r", bit0_value)

            converting_every_integer_to_binary = '{:06b}'.format(bit0_value)

        logger.dbg("every%rchar --> %rbit\n\t%r --> %r\n", len(tmpstr), len(converting_every_integer_to_binary), tmpstr, converting_every_integer_to_binary)

        bitstring += converting_every_integer_to_binary
        data = data[2:]

    logger.dbg("byte mode + char cnt + data=\n\t%r\n", convert_every8bit_str(bitstring))

    return bitstring

def _encode_numeric_mode(data):
    logger.dbg("\n\tdata=%r\n", data)

    #
    # 1. 添加 模式指示符:
    # 数字 mode: 0001
    #
    bitstring = '0001'
    logger.dbg("Mode=\n\t%r\n", bitstring)

    #
    # 2. 添加 字符数指示符;
    # Character count: 10 bits.
    #
    bitstring += '{:010b}'.format(len(data))
    logger.dbg("Mode + char CNT=\n\t%r\n", convert_every8bit_str(bitstring))

    #
    # 3. 在数字模式下, 每3个数据 分为一组, 然后 分别被编码成10bit的二进制数;
    #
    tmpstr = data
    while tmpstr:
        if len(tmpstr) >= 3:
            converting_every_integer_to_binary = '{:010b}'.format(int(tmpstr[:3]))
        elif len(tmpstr) == 2:
            converting_every_integer_to_binary = '{:07b}'.format(int(tmpstr))
        elif len(tmpstr) == 1:
            converting_every_integer_to_binary = '{:04b}'.format(int(tmpstr))

        logger.dbg("every%rint --> %rbit=\n\t%r --> %r\n", len(tmpstr[:3]), len(converting_every_integer_to_binary),  tmpstr[:3], converting_every_integer_to_binary)

        tmpstr = tmpstr[3:]
        bitstring += converting_every_integer_to_binary

    logger.dbg("mode + char cnt + data=\n\t%r\n", convert_every8bit_str(bitstring))

    return bitstring


def _encode(data):
    '''
    编码 输入数据,
        返回 一维 整数矩阵 [ 模式指示符 + 字数指示符 + 数据内容 + 终止符 + 容错码 ]

    Encode the input data stream.
    Add mode prefix, encode data using ISO-8859-1,
    group data, add padding suffix, and call RS encoding method.
    '''
    logger.dbg("input-data, len=%r, [%r]", len(data), data)

    bitstring = ''
    #
    # 检测输入的数据是否超过V1-L byte mode 的最大编码长度17,
    #       如果超过就抛出异常
    #
    if len(data) > 17:
        raise CapacityOverflowException(
                'Error: Version 1 QR code[binary mode] encodes no more than 17 characters.')

    ###########################################
    #                 start                   #
    ###########################################

    # 保存 已经解码的字符 个数;
    has_encode_cnt = 0

    while(True):
        # 获取需要编码的数据 起始地址
        need_encode_data = data[has_encode_cnt:]
        logger.dbg("data-len=%d, data=%r", len(need_encode_data), need_encode_data)
        
        # 设置 初始的flag;
        curCharMode = None
        start_index = 0

        # 判断 当前字符 是否 属于 数字模式:
        if curCharMode == None:
            while(True):
                is_numeric_mode_char = need_encode_data[start_index]
                if ord(is_numeric_mode_char) >= ord('0') and ord(is_numeric_mode_char) <= ord('9'):
                    curCharMode = numeric_mode
                    start_index += 1
                else:
                    break

        # 判断 当前字符 是否 属于 字母数字模式:
        if curCharMode == None:
            while(True):
                is_alphanumeric_mode_char = need_encode_data[start_index]

                # 大写字母 和 数字
                if ( ord(is_alphanumeric_mode_char) >= ord('A') and ord(is_alphanumeric_mode_char) <= ord('Z')) or (ord(is_alphanumeric_mode_char) >= ord('0') and ord(is_alphanumeric_mode_char) <= ord('9')):
                    curCharMode = alphanumeric_mode
                    start_index += 1
                else:
                    break

        # 剩下的 所有字符 使用 字节模式 编码:
        if curCharMode == None:
            curCharMode = byte_mode
            start_index = len(need_encode_data)

        if curCharMode != None:
            # 获取 需要使用 当前模式 来编码的 字符串;
            tmp_encode_data = data[has_encode_cnt:has_encode_cnt+start_index]

            if curCharMode == numeric_mode:
                bitstring += _encode_numeric_mode(tmp_encode_data)
            elif curCharMode == alphanumeric_mode:
                bitstring += _encode_alphanumeric_mode(tmp_encode_data)
            else :
                bitstring += _encode_byte_mode(tmp_encode_data)

        logger.dbg("Mode + char-Cnt + Data=\n\t%r\n", convert_every8bit_str(bitstring))

        # 统计已经 编码的数据
        has_encode_cnt += start_index

        # 判断 所有的数据 是否 已经全部编码 
        if has_encode_cnt == len(data):
            break

            '''
            sys.exit()
            '''
    if curEncodeMode == numeric_mode:#数字模式
        bitstring = _encode_numeric_mode(data)
    elif curEncodeMode == alphanumeric_mode:# 字母数字模式
        bitstring = _encode_alphanumeric_mode(data)
    elif curEncodeMode == mixed_mode:# 混合模式
        pass
    else :# 字节模式
        bitstring = _encode_byte_mode(data)
        
    ###########################################
    #                 end                     #
    ###########################################

    #
    # 4. 添加终止符 [0000],
    #
    bitstring += '0000'
    logger.dbg("mode + char cnt + data + terminater=\n\t%r\n", convert_every8bit_str(bitstring))

    #
    # 5. 每8bit分为一组, 如果尾部数据不足8bit，则在尾部 填充0:
    #
    tmpstr = bitstring
    last_str = ''
    while tmpstr:
        last_str = tmpstr[:8]
        tmpstr = tmpstr[8:]

    bitstring += '0'*(8-len(last_str))
    #logger.dbg("last-str, len=%r, data=%r, append-0=%r", len(last_str), last_str, last_str+'0'*(8-len(last_str)))
    logger.dbg("byte mode + char cnt + data + terminater + append str=\n\t%r\n", convert_every8bit_str(bitstring))

    res = list()
    #
    # 6. 把 每8位 二进制数据 转换为 整数;
    #
    while bitstring:
        res.append(int(bitstring[:8], 2))
        bitstring = bitstring[8:]
    logger.dbg("convert byte to int=\n\t%r\n", res)

    #
    # 7. 如果编码后的数据不足版本及纠错级别的最大容量,
    #       则在尾部补充 "11101100" 和 "00010001"
    #
    # V1-L 的 数据码字数: 19个
    #
    while len(res) < 19:
        res.append(int('11101100', 2))
        res.append(int('00010001', 2))

    #
    # 8. 截取 前19个字符
    #
    res = res[:19]

    logger.dbg("value:1~19, data=\n\t%r\n", res)

    #
    # 9. 添加 RS容错码;
    #
    # V1-L 的 容错码字数: 7个
    #
    return _rsEncode(res, 7)

def _fillByte(byte, downwards=False):
    '''
    把 1个字节数据 转换为 一个2x4的矩形,
        也就是 实现单个字节的填充, 如下:

        例如: 一个 8bit的二进制字 '0b 0010 1101' 左边是高位, 右边是低位,

        在内存里, 正好相反, 左边是低位, 右边是高位, 如下:
        ---------------------------------------------------------
        | bit0 | bit1 | bit2 | bit3 | bit4 | bit5 | bit6 | bit7 |
        ---------------------------------------------------------
        |   0  |   0  |   1  |   0  |   1  |   1  |  0   |   1  |
        ---------------------------------------------------------

        Upwards模式:
            把 字节(byte) 里的 bit 7 和 bit 6, 放在 矩形的 (y0, x0) 和 (y0, x1),
            以此类推, 把 bit1 和 bit0, 放在 矩形的 (y3, x0) 和 (y3, x1);

    --------------------------------

          y  x       Upwards                Downwards
         (0, 0) ---------------          ---------------    ----> X轴
                | bit7 | bit6 |  ^       | bit1 | bit0 |    
         (1, 0) ---------------  |       ---------------  | 
                | bit5 | bit4 |  |       | bit3 | bit2 |  |  
         (2, 0) ---------------  |       ---------------  | 
                | bit3 | bit2 |  |       | bit5 | bit4 |  |  
         (3, 0) ---------------  |       ---------------  | 
                | bit1 | bit0 |  |       | bit7 | bit6 |  v  
         (4, 0) ---------------          ---------------    

                |
                |
                v
               Y轴

    Fill a byte into a 2 by 4 matrix upwards,
    unless specified downwards.
    '''
    bytestr = '{:08b}'.format(byte)
    res = [[0, 0], [0, 0], [0, 0], [0, 0]]

    for y in range(8):
        i = x = y
        if DARK_IS_1:
            res[y/2][x%2] = int(bytestr[7-i])
        else:
            res[y/2][x%2] = not int(bytestr[7-i])

    #logger.dbg("a-byte-len=%d, data=%r", len(res), res)
    if downwards:
        res = res[::-1]
        #logger.dbg("revers:len=%d, data=%r", len(res), res)

    #print "\n"
    return res

def _fillData(bitstream):
    '''
    将 数据填充到 一个临时的 矩阵模板中, 一个字节 占 8个点位,

        V1-L: 一共有26个字节(encode 19 + rscode 7)

    Fill the encoded data into the template QR code matrix
    '''

    #
    # 1. 先获取一份 已经填充了功能图形的 的 矩形(数据结构)
    #
    res = copy.deepcopy(_ver1)

    #
    # 2.1. 填充 第一部分数据, 一共15个数据, 0~14
    #    每一个数据是8个bit, 每一个bit占用 一个块;
    #
    for i in range(15):
        res = _matCp(_fillByte(bitstream[i], (i/3)%2!=0),
            res,
            21-4*((i%3-1)*(-1)**((i/3)%2)+2), #先向上, 后向下
            21-2*(i/3+1)) #向左 移动2个单位:块

    # 2.2. 填充 第二部分数据, 1个, 被 水平定位图像 分割 为 两部分;
    tmp = _fillByte(bitstream[15])
    res = _matCp(tmp[2:], res, 7, 11)
    res = _matCp(tmp[:2], res, 4, 11)

    #logger.dbg("15, len=%d, data=%r", len(tmp), tmp)

    # 2.3. 填充 第三部分数据, 1个;
    tmp = _fillByte(bitstream[16])
    res = _matCp(tmp, res, 0, 11)

    # 2.4. 填充 第四部分数据, 1个;
    tmp = _fillByte(bitstream[17], True)
    res = _matCp(tmp, res, 0, 9)

    # 2.5. 填充 第五部分数据, 1个, 被 水平定位图像 分割 为 两部分;
    tmp = _fillByte(bitstream[18], True)
    res = _matCp(tmp[:2], res, 4, 9)
    res = _matCp(tmp[2:], res, 7, 9)

    # 2.6. 填充 第六部分数据, 3个;
    for i in range(3):
        res = _matCp(_fillByte(bitstream[19+i], True),
            res, 9+4*i, 9)

    # 2.7. 填充 第七部分数据, 1个;
    tmp = _fillByte(bitstream[22])
    res = _matCp(tmp, res, 9, 7)

    # 2.8. 填充 第八部分数据, 最后 3个;
    for i in range(3):
        res = _matCp(_fillByte(bitstream[23+i], i%2==0),
            res, 9, 4-2*i)

    return res

def _matXor(dataMat, dataMaskMat):
    '''
    数据 和 掩码 按位 进行 [异或]

    Matrix-wise xor.
    Dark xor dark -> light
    Light xor light -> light
    Dark xor light -> dark
    Light xor dark -> dark
    '''
    res = [[_LIGHT for i in range(len(dataMat[0]))] for j in range(len(dataMat))]

    for j in range(len(dataMat)):
        for i in range(len(dataMat[0])):
            if DARK_IS_1:
                res[j][i] = int(dataMat[j][i] ^ dataMaskMat[j][i])
            else:
                res[j][i] = int(dataMat[j][i] == dataMaskMat[j][i])

    #logger.dbg("old  mat, len=%r, data=%r", len(dataMat), dataMat)
    #logger.dbg("mask mat, len=%r, data=%r", len(dataMaskMat), dataMaskMat)
    #logger.dbg("last mat, len=%r, data=%r", len(res), res)

    return res

def qrcode(data, width=210, filename='QR-code'):
    '''
    创建 二维码

    Module public interface
    '''
    try:
        if curEncodeMode == numeric_mode:#数字模式
            logger.dbg("use numeric mode")
            filename += "-numeric.png"
        elif curEncodeMode == alphanumeric_mode:# 字母数字模式
            logger.dbg("use alphanumeric mode")
            filename += "-alphanumeric.png"
        elif curEncodeMode == mixed_mode:# 混合模式
            logger.dbg("use mixed mode")
            filename += "-mixed.png"
        elif curEncodeMode == chinese_mode:# 汉字模式
            logger.dbg("use chinese mode")
        else :# 默认, 字节模式
            logger.dbg("use byte mode")
            filename += "-byte.png"

        _genImage(_genBitmap(_encode(data)), width, filename)
    except Exception, e:
        print e
        raise e
    logger.dbg("Create [%r] QR-code ok.", data)

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

#008-test
_genImage(_ver1, 210, 'test_ver1_func.png')

'''
qrcode('1986AZ5gh')

