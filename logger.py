#!/usr/bin/python
# coding: utf-8

'''
使用方法:

import logger

logger.ENABLE_DEBUG = False

logger.DBG('hi nexgo')

'''

import sys
import time

MY_TAG = 'wanghai_debug'

'''
控制log输出开关;
'''
ENABLE_DEBUG = True

def MY_INFO(format_str='', *args):
    output_info(format_str % args)

def INFO(format_str='', *args):
    output_info(format_str % args)

def my_info(format_str='', *args):
    output_info(format_str % args)

def info(format_str='', *args):
    output_info(format_str % args)

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

    if print_info == '':
        print "%s, Time=[%s], file=[%s], @%s, line=%s" % \
                (MY_TAG, time.strftime("%Y-%m-%d, %H:%M:%S"), f.f_code.co_filename, f.f_code.co_name, f.f_lineno)

        #print "file=[%s]" % (__file__)
    else:
        print "%s, file=[%s], @%s, line=%s, %s" % \
                (MY_TAG, f.f_code.co_filename, f.f_code.co_name, f.f_lineno, print_info)





