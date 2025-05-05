#!/usr/bin/env python
#coding:utf8
#将log文件中的最终准确率结果输出为result.log
import os
import re
import sys
regtxt = r'.+?\.log' #扫描对象为txt文件.
regcontent = r'AccuracyOverall' #列出内容含有'what is your name'的文件
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a",encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


type = sys.getfilesystemencoding()
sys.stdout = Logger(r"results.log")
class FileException(Exception):
    pass

def getdirlist(filepath):
    """获取目录下所有的文件."""

    txtlist = [] #文件集合.
    txtre = re.compile(regtxt)
    needfile = [] #存放结果.
    for parent, listdir, listfile in os.walk(filepath):
        for files in listfile:
            #获取所有文件.
            istxt = re.findall(txtre, files)
            filecontext = os.path.join(parent, files)
            #获取非空的文件.
            if istxt :
                txtlist.append(filecontext)
                #将所有的数据存放到needfile中.
                needfile.append(readfile(filecontext,files)) 

    if needfile == []:
        raise FileException("no file can be find!")
    else:
        getvalidata(needfile)
        #print(validatedata)
        #print('total file %s , validate file %s.' %(len(txtlist),len(validatedata)))

def getvalidata(filelist=[]):
    """过滤集合中空的元素."""

    valifile = []
    for fp in filelist:
        if fp != None:
            valifile.append(fp)
    return valifile
'''
def readfile(filepath,files):
    """通过正则匹配文本中内容，并返回文本."""

    flag = False
    contentre = re.compile(regcontent)
    fp = open(filepath, 'r',encoding='utf-8')
    #lines = fp.readlines()
    #flines = len(lines)
    #逐行匹配数据.
    for line in fp:
    #for i in range(flines): 
        iscontent = re.findall(contentre, line) 
        if iscontent:
            fp.close()
            new_files=files[:files.rfind(".")]
            for i in range(0,12):
                if new_files==dataset[i]:
                    print(str(i)+" "+new_files)
                    print(line)
                    return filepath
'''

def readfile(filepath,files):
    """通过正则匹配文本中内容，并返回文本."""

    flag = False
    contentre = re.compile(regcontent)
    fp = open(filepath, 'r',encoding='utf-8')
    #lines = fp.readlines()
    #flines = len(lines)
    #逐行匹配数据.
    for line in fp:
    #for i in range(flines): 
        iscontent = re.findall(contentre, line) 
        if iscontent:
            #new_files=files[:files.rfind(".")]
            #line=line[:line.rfind(".")]
            line=line[:line.rfind("%")]
            line1=line[:line.rfind(" A")]
            line2=line[line.rfind("l ")+1:]
            new_line=line1+line2
            print(new_line)
    return filepath

if __name__ == "__main__":
    getdirlist('.')