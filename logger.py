#!/usr/bin/python
# coding: utf-8

import sys
import time

MY_TAG = 'wanghai_debug'
def MY_INFO(print_info=''):
    #use_exception = ''

    #if isset('use_exception'):
    if vars().has_key('use_exception'):
        #print "use exception."
        try:
            raise Exception
        except:
            f = sys.exc_info()[2].tb_frame.f_back
    else:
        #print "use sys."
        f = sys._getframe().f_back

    if print_info == '':
        print "%s, Time=[%s], file=[%s], @%s, line=%s" % \
                (MY_TAG, time.strftime("%Y-%m-%d, %H:%M:%S"), f.f_code.co_filename, f.f_code.co_name, f.f_lineno)

        #print "file=[%s]" % (__file__)
    else:
        print "%s, file=[%s], @%s, line=%s, %s" % \
                (MY_TAG, f.f_code.co_filename, f.f_code.co_name, f.f_lineno, print_info)

