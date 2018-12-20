#!/usr/bin/python
# coding: utf-8

import sys
import time

'''
使用方法:

import logger

logger.ENABLE_DEBUG = False

logger.DBG('hi nexgo')

'''

MY_TAG = 'wanghai_debug'

###############################################
'''
控制log的 输出开关;
'''
ENABLE_DEBUG = True

'''
控制相关的log输出开关
'''
PRINT_TAG=False

PRINT_BUILDTIME=False

PRINT_FILENAME=False

###############################################
def MY_INFO(format_str='', *args):
    output_info(format_str % args)

def INFO(format_str='', *args):
    output_info(format_str % args)

def my_info(format_str='', *args):
    output_info(format_str % args)

def info(format_str='', *args):
    output_info(format_str % args)

###############################################
def MY_DEBUG(print_info=''):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(print_info)

def DEBUG(print_info=''):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(print_info)

def DBG(print_info=''):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(print_info)

def dbg(format_str='', *args):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(format_str % args)

def my_debug(print_info=''):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(print_info)

def debug(print_info=''):
    if isset('ENABLE_DEBUG') and ENABLE_DEBUG:
        output_info(print_info)

###############################################
def isset(v):
   try :
     type (eval(v))
   except :
     return False
   else :
     return True

def output_info(print_info=''):
    #use_exception = ''

    if vars().has_key('use_exception'):
        #print "use exception."
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back.f_back
    else:
        #print "use sys."
        f = sys._getframe().f_back.f_back

    output_str = ''

    if PRINT_TAG:
        output_str += "%s, " % (MY_TAG)

    if PRINT_BUILDTIME:
        output_str += "Time=[%s], " % (time.strftime("%Y-%m-%d, %H:%M:%S"))

    if PRINT_FILENAME:
        output_str += "file=[%s], " % (f.f_code.co_filename)
        #output_str += "file=[%s], " % (__file__)

    if print_info == '':
        output_str += "@%s, line=%s" % (f.f_code.co_name, f.f_lineno)
    else:
        output_str += "@%s, line=%s, %s" % (f.f_code.co_name, f.f_lineno, print_info)

    print output_str


