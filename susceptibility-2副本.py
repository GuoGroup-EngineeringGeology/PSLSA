from PyQt5 import uic
from PyQt5.QtWidgets import  QMessageBox
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
import shutil
import sys
import base64
#from github import Github
import os
from PyQt5.QtGui import QIcon
import numpy as np
from PIL import Image

from osgeo import gdal,ogr
import sys
import os
import xlwt
#import ogr
import pandas as pd
import math
import time
import random
import configparser
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['Kaiti']
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import sklearn
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing                       #预处理模块
from sklearn.impute import SimpleImputer                #数据插补模块
from sklearn.preprocessing import OneHotEncoder         #OneHotEncoder编码模块
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler        #标准化模块
from sklearn.preprocessing import MinMaxScaler          #01归一化模块
from sklearn.pipeline import Pipeline                   #流程式处理模块
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
#import jenkspy       #自然断点法
from datetime import datetime
from skimage import io

from rpy2.robjects.packages import importr
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
pandas2ri.activate()
C50 = importr('C50')

import warnings
warnings.filterwarnings("ignore")

from PyQt5.QtGui import QPixmap
app = QApplication([])
class Stats(QMainWindow):

    def __init__(self):

        super().__init__()
        # self.initUI()
        # 从文件中加载UI定义
        self.ui = uic.loadUi("滑坡易发性评价ui界面0.ui")

        #设置图标 及 主标题
        self.ui.setWindowIcon(QIcon('图片.ico'))
        self.ui.setWindowTitle('滑坡自动化易发性评价系统')
        # 信号和槽进行连接、初始化变量
        self.disaster_number = 79
        self.ui.pushButton.clicked.connect(self.read_zaidiannfilepath)#高程、坡度、坡向、剖面曲率、
        self.ui.pushButton_10.clicked.connect(self.read_zaidiannfilepath2)  # 高程、坡度、坡向、剖面曲率、
        self.ui.pushButton_17.clicked.connect(self.yifaxing_pre)

        self.lisan_dict = {}
        #self.fa_inpath.update(xx_f)  #存储额外因子的原始输入路径，输出路径有一套规则，不用存储
        self.all_factor_filepath=["","","","","","","","","",""] # 分别按顺序存储：高程、坡度、坡向、剖面曲率、地层岩性、降雨量、NDVI，距河流距离;
                                                         # 坡度、坡向、剖面曲率为高程文件得来，若勾选对话框设置为高程路径，否则为"" ，"'为不使用
        self.all_factor_name = ["高程","坡度","坡向","剖面曲率","平面曲率","粗糙度","地层岩性","降雨量","NDVI","距河流距离"]
        self.basic_fac = ["altitude","slope","aspect","profile curvature","plan curvature","SDS","lithology","rainfall","NDVI","dis_to_river"]
        self.factor_name = []
        self.fa_inpath = {}

        self.ui.pushButton_2.clicked.connect(self.read_filepath_0123)#高程、坡度、坡向、剖面曲率、（平面曲率、粗糙度）
        self.ui.pushButton_3.clicked.connect(self.read_filepath_4)#地层岩性
        self.ui.pushButton_4.clicked.connect(self.read_filepath_5)#降雨量
        self.ui.pushButton_5.clicked.connect(self.read_filepath_6)#NDVI
        self.ui.pushButton_6.clicked.connect(self.read_filepath_7)#距河流距离
        self.ui.pushButton_11.clicked.connect(self.attend_factor)  # 距河流距离
        self.ui.pushButton_13.clicked.connect(self.read_LULC)  # LULC
        self.ui.pushButton_12.clicked.connect(self.read_gouzao)  # 距构造距离

        self.ui.pushButton_32.clicked.connect(self.show_cutoff_point)
        self.ui.pushButton_18.clicked.connect(self.factor_importance)
        self.ui.pushButton_33.clicked.connect(self.checkVIF_new)


        self.ui.pushButton_16.clicked.connect(self.model_use_showroc)
        self.ui.pushButton_9.clicked.connect(self.show_zhuantitupian)

        self.modelname_list=["LR_model","MLP_mdoel","SVM_model","XGBOOST_mdoel","RF_model","C5.0_model"]
        #self.ui.comboBox_3.addItem(self.modelname_list[0])
        #self.ui.comboBox_3.addItem(self.modelname_list[1])
        #self.ui.comboBox_3.addItem(self.modelname_list[2])
        #self.ui.comboBox_3.addItem(self.modelname_list[3])
        self.factor_dic={'高程':r'.\data\re_altitude.tif','坡度':r'.\data\reslope.tif','坡向':r'.\data\re_aspecct.tif',
                         '剖面曲率':r'.\data\re_curve.tif','降雨量':r'.\data\re_rainfall.tif','NDVI':r'.\data\re_ndvi.tif',
                         '距河流距离':r'.\data\re_dr.tif','地层岩性':r'.\data\re_yanzu.tif',
                         '粗糙度':r'.\data\surface roughness.tif','平面曲率':r'.\data\plancurve.tif'}


    # 因子分级按钮
        #self.modellist = os.listdir(".\\model")
        #for hejun in  self.modellist:
            #self.ui.comboBox_3.addItem(hejun[:-2])
            #self.ui.comboBox_2.addItem(hejun[:-2])

        self.ui.pushButton_7.clicked.connect(self.factor_lisan)  # 距河流距离
    #数据转换按钮
        self.ui.pushButton_8.clicked.connect(self.cal_fr)
        self.ui.pushButton_15.clicked.connect(self.all_model_train)
        self.modelpara=[0,0,0,0,0,0]

        self.lrmodel=0
        self.dcgzjmodel=0
        self.svmmodel=0
        self.xgboostmodel=0
        self.lrmodelpara=0
        self.dcgzjmodelpara=0
        self.svmmodelpara=0
        self.xgboostmodelpara=0
    # tiff_path=r"E:\pycharm project\python-gis\python-gis\avetry1.tif"
        # #im = io.imread(tiff_path)
        # pixmap = QPixmap(tiff_path)
        # self.ui.label_17.setPixmap(pixmap)
##上面是初始化

    def read_zaidiannfilepath(self):
        #显示选择文件路径
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.lujing1 = openfile_name1[0]
            self.ui.lineEdit.setText(self.lujing1)

    def read_zaidiannfilepath2(self):
        # 显示选择文件路径
        openfile_name2 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name2[0] != "":
            self.lujing2 = openfile_name2[0]
            self.ui.lineEdit_14.setText(self.lujing2)
        #openfile_name2 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        #if openfile_name2[0] != "":
            #self.lujing2 = openfile_name2[0]
            #self.ui.lineEdit.setText(self.lujing1+";"+self.lujing2)


        self.disaster_data, self.disaster_number = self.disaster_point(self.lujing1, self.lujing2)
        self.data_y = self.random_non_disaster_point(self.lujing2, self.disaster_number ,self.disaster_data )
        #print(self.data_y)
        print(len(self.data_y))

        self.ui.textBrowser.append("文件1"+self.lujing1)  # 在界面的框里显示一下“高程”
        self.ui.textBrowser.append("")  # 在界面的框里添加一个空行
        self.ui.textBrowser.append("文件2"+self.lujing2)  # 在界面的框里显示一下“高程”
        self.ui.textBrowser.append("")  # 在界面的框里添加一个空行

#函数1，用来写入高程、坡度、破面、剖面曲率 文件路径 及 进行数据转换
    def read_filepath_0123(self):  #1234代表高程、坡度、破面、剖面曲率的顺序(平面曲率，粗糙度)
        openfile_name1 = QFileDialog.getOpenFileName(self,'选择文件','','')

        if openfile_name1[0]!="":
            self.filepath_0123=openfile_name1[0]
            # 分别检测一下 高程坡度、破面、剖面曲率 是否选中

            self.ui.lineEdit_2.setText(self.filepath_0123)


            # 计算有数值栅格得范围
            tif_read = self.read_tif_file(self.filepath_0123)
            data = tif_read[2]
            self.scope = np.where(data < -300, np.nan, data)  # self.scope 是一个数据有数据的地方不变，无数据的地方为np.nan
            # 后面输出的tif，都要经过这个数组作为掩膜筛选
            # 方法，如计算出的slope要转化为 slope1 = np.where(np.isnan(self.scope),np.nan,slope)

            if self.ui.checkBox.isChecked() == True:

                self.ui.comboBox_4.addItem('altitude')   #因子名称

                self.all_factor_filepath[0]=self.filepath_0123    #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_7
                #self.ui.lineEdit_7.setEnabled(True)   #高程
                self.Cal_height(self.filepath_0123, 30)

                self.ui.lineEdit_59.setEnabled(True)
                self.factor_name.append('高程')

             # 高程函数（self.filepath_0123 ）
            if self.ui.checkBox_2.isChecked()==True:
                self.ui.comboBox_4.addItem('slope')
                self.all_factor_filepath[1] = self.filepath_0123  #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_8
                #self.ui.lineEdit_8.setEnabled(True)    #坡度
                self.Cal_slpoe(self.filepath_0123, 30)
                self.factor_name.append('坡度')

            if self.ui.checkBox_3.isChecked() == True:
                self.ui.comboBox_4.addItem('aspect')
                self.all_factor_filepath[2] = self.filepath_0123  #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_9
                #self.ui.lineEdit_9.setEnabled(True)  #坡向
                self.cal_aspect(self.filepath_0123, 30)
                self.factor_name.append('坡向')


            if self.ui.checkBox_4.isChecked()==True:
                self.ui.comboBox_4.addItem('profile curvature')
                self.all_factor_filepath[3] = self.filepath_0123  #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_10
                #self.ui.lineEdit_10.setEnabled(True)    #剖面曲率
                self.cal_cracurve(self.filepath_0123, 30)
                self.factor_name.append('剖面曲率')

            if self.ui.checkBox_16.isChecked()==True:
                self.ui.comboBox_4.addItem('plan curvature')
                self.all_factor_filepath[4] = self.filepath_0123  #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_10
                #self.ui.lineEdit_26.setEnabled(True)   #平面曲率
                self.cal_plancurve(self.filepath_0123, 30)
                self.factor_name.append('平面曲率')

            if self.ui.checkBox_14.isChecked()==True:
                self.ui.comboBox_4.addItem('SDS')
                self.all_factor_filepath[5] = self.filepath_0123  #把这个路径放到列表里
                # 因子分级及编码对其进行显示
                # lineEdit_10
                #self.ui.lineEdit_39.setEnabled(True)    #粗糙度
                self.SDS(self.filepath_0123, 30)
                self.factor_name.append('粗糙度')

    # 底层岩性
    def read_filepath_4(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('lithology')
            self.filepath_4 = openfile_name1[0]
            self.all_factor_filepath[6] = self.filepath_4  # 把这个 地层岩性 路径放到列表里
            self.ui.lineEdit_3.setText(self.filepath_4)
            # 因子分级及编码对其进行显示
            # lineEdit_42
            #self.ui.lineEdit_42.setEnabled(True)
            # 添加 地层岩性 数据转换函数或者高程数据读取函数
            self.write_yanzu(self.filepath_4)
            self.factor_name.append('地层岩性')

    def read_filepath_5(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('rainfall')
            self.filepath_5 = openfile_name1[0]
            self.all_factor_filepath[7] = self.filepath_5  # 把这个 降雨量 路径放到列表里
            self.ui.lineEdit_4.setText(self.filepath_5)
            # 因子分级及编码对其进行显示
            # lineEdit_12
            #self.ui.lineEdit_12.setEnabled(True)
            # 添加 降雨量 数据转换函数或者高程数据读取函数
            self.write_rainfall(self.filepath_5)
            self.factor_name.append('降雨量')


    def read_filepath_6(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('NDVI')
            self.filepath_6 = openfile_name1[0]
            self.all_factor_filepath[8] = self.filepath_6  # 把这个 ndvi 路径放到列表里
            self.ui.lineEdit_5.setText(self.filepath_6)
            # 因子分级及编码对其进行显示
            # lineEdit_11
            self.ui.lineEdit_59.setEnabled(True)
            # 添加 ndvi 数据转换函数或者高程数据读取函数
            self.write_ndvi(self.filepath_6)
            self.factor_name.append('NDVI')


    def read_filepath_7(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('dis_to_river')
            self.filepath_7 = openfile_name1[0]
            self.all_factor_filepath[9] = self.filepath_7  # 把这个 距河流距离 路径放到列表里
            self.ui.lineEdit_6.setText(self.filepath_7)
            # 因子分级及编码对其进行显示
            self.ui.lineEdit_59.setEnabled(True)
            # 添加 距河流距离 数据转换函数或者高程数据读取函数
            self.write_river(self.filepath_7)
            self.factor_name.append('距河流距离')

    def read_LULC(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('LULC')
            self.filepath_lulc = openfile_name1[0]
            #self.all_factor_filepath[10] = self.filepath_lulc  # 把这个 距河流距离 路径放到列表里
            self.ui.lineEdit_8.setText(self.filepath_lulc)
            # 因子分级及编码对其进行显示
            self.ui.lineEdit_59.setEnabled(True)
            # 添加 距河流距离 数据转换函数或者高程数据读取函数
            #self.write_river(self.filepath_lulc)
            self.factor_name.append('LULC')
            #self.factor_name.append(F1)  # 因子名称
            #stf = str(F1)
            #outpath = r'.\data\%s.tif' % 'LULC'
            #filepath_a = openfile_name1[0]
            xx_f = {'LULC': openfile_name1[0]}
            self.fa_inpath.update(xx_f)

    def read_gouzao(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        if openfile_name1[0] != "":
            self.ui.comboBox_4.addItem('dis_to_falut')
            self.filepath_gaozao = openfile_name1[0]
            #self.all_factor_filepath[11] = self.filepath_gaozao  # 把这个 距河流距离 路径放到列表里
            self.ui.lineEdit_7.setText(self.filepath_gaozao)
            # 因子分级及编码对其进行显示
            self.ui.lineEdit_59.setEnabled(True)
            # 添加 距河流距离 数据转换函数或者高程数据读取函数
            #self.write_river(self.filepath_gaozao)
            self.factor_name.append('距构造距离')
            xx_f = {'距构造距离': openfile_name1[0]}
            self.fa_inpath.update(xx_f)


    def attend_factor(self):
        openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
        self.ui.lineEdit_60.setEnabled(True)
        if self.ui.lineEdit_60 != "":
            #openfile_name1 = QFileDialog.getOpenFileName(self, '选择文件', '', '')
            F1 = self.ui.lineEdit_60.text()
            self.factor_name.append(F1)   #因子名称
            stf = str(F1)
            outpath = r'.\data\%s.tif' % stf
            self.ui.comboBox_4.addItem(stf)
            filepath_a = openfile_name1[0]
            xx_f = {F1: openfile_name1[0]}
            self.fa_inpath.update(xx_f)
            #self.all_factor_filepath[9] = self.filepath_7  # 把这个 距河流距离 路径放到列表里
            self.ui.lineEdit_61.setText(filepath_a)
            self.ui.textBrowser_4.append(stf + str(openfile_name1[0]))
            self.factor_name.append(stf)


    def show_cutoff_point(self):
        cur_text = self.ui.comboBox_4.currentText()   #选择的因子名

        '''lian_dic = ['坡向','地层岩性']
        if cur_text in lian_dic:
            self.ui.lineEdit_59.setEnabled(False)
        else:
            self.ui.lineEdit_59.setEnabled(True)'''
        xx_f = {cur_text:[self.ui.lineEdit_59.text()]}
        self.lisan_dict.update(xx_f)
        print(self.lisan_dict)
        self.ui.textBrowser_3.append(cur_text+str([self.ui.lineEdit_59.text()]))
        #self.ui.lineEdit_59.text = ''


    def factor_lisan(self):    #点击进行分级 执行这个代码
        #字典形式用于存储输入的分级标准，如果为空则表示这个因素不需要执行
        print(self.lisan_dict)
        '''self.lisan_dict={"高程":"","坡度":"","坡向":"","平面曲率":"","剖面曲率":"","粗糙度":"","地层岩性":"","降雨量":"","NDVI":"",
                         "距河流距离":""}'''
        selected_factors = list(self.lisan_dict.keys())  #返回类似于['name', 'age', 'birthday', 'sex']
        for fs in range(len(selected_factors)):
            self.ui.textBrowser_2.append(selected_factors[fs] + str('  分级已完成'))  # 在界面的框里显示一下“高程”
            self.ui.textBrowser_2.append("")  # 在界面的框里添加一个空行
            if selected_factors[fs] in self.basic_fac:
                if selected_factors[fs] == 'altitude':

                    s = self.lisan_dict["altitude"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/altitude.tif'
                    self.Cac_altitude(read_filepath, s1)
                if selected_factors[fs] == 'slope':
                    s = self.lisan_dict["slope"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/slope.tif'
                    self.Cac_slpoe(read_filepath, s1, 30)

                if selected_factors[fs] == 'aspect':
                    read_filepath = r'./data/aspect.tif'
                    self.Cac_Aspect(read_filepath)
                if selected_factors[fs] == 'plan curvature':
                    s = self.lisan_dict["plan curvature"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/plancurve.tif'
                    replancurve_path = r'./data/re_plancure.tif'
                    # self.Cac_curvature(read_filepath, s1)
                    self.divise_factor(read_filepath, s1, replancurve_path)
                if selected_factors[fs] == 'profile curvature':
                    s = self.lisan_dict["profile curvature"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/curve.tif'
                    self.Cac_curvature(read_filepath, s1)

                if selected_factors[fs] == 'SDS':
                    s = self.lisan_dict["SDS"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/surface roughness.tif'
                    replancurve_path = r'./data/re_sr.tif'
                    # self.Cac_curvature(read_filepath, s1)
                    self.divise_factor(read_filepath, s1, replancurve_path)
                if selected_factors[fs] == 'lithology':
                    read_filepath = r'./data/yanzu.tif'
                    self.write_yanzu2(read_filepath)
                if selected_factors[fs] == 'rainfall':
                    s = self.lisan_dict["rainfall"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/rainfall.tif'
                    self.Cac_Rainfall_distribution(read_filepath, s1)
                if selected_factors[fs] == 'NDVI':
                    s = self.lisan_dict["NDVI"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/ndvi.tif'
                    self.Cac_NDVI(read_filepath, s1)
                if selected_factors[fs] == 'dis_to_river':
                    s = self.lisan_dict["dis_to_river"][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = r'./data/dis_river.tif'
                    out_path = r'./data/re_dr.tif'
                    self.distence_to_river(read_filepath, s1, out_path)

            #讨论离散和连续因子
            else:
                if self.lisan_dict[selected_factors[fs]] != ['']:
                    print(self.lisan_dict[selected_factors[fs]])
                    s = self.lisan_dict[selected_factors[fs]][0].split(',')
                    s1 = []
                    for i in range(len(s)):
                        s1.append(float(s[i]))
                    print(type(s1))
                    print(s1)
                    read_filepath = self.fa_inpath[selected_factors[fs]]
                    replancurve_path = r'./data/re_'+selected_factors[fs]+'.tif'
                    # self.Cac_curvature(read_filepath, s1)
                    self.divise_factor(read_filepath, s1, replancurve_path)
                else:
                    read_filepath = self.fa_inpath[selected_factors[fs]]
                    #因子的输入路劲，self，fa_inpath{factor,inpath},self.lisan_dict[fs]={factor，区间}
                    print(read_filepath)
                    replancurve_path = r'./data/re_' + selected_factors[fs] + '.tif'
                    tif_read = self.read_tif_file(read_filepath)
                    data = tif_read[2]
                    data = np.where(np.isnan(data), 0, data)
                    self.preit_to_tif(data,read_filepath,replancurve_path)

    def cal_fr(self):   #点击数据编码执行这个函数
        #self.lisan_dict={'factor':[cutoff-points]}
        #self.lisan_dict#字典，键名为 名称，值为列表
        fx = r'.\data\altitude.tif'
        #ly = r'.\data\landslide.csv'
        tem_dat = self.read_tif_file(fx)[2]
        #tem_l = pd.read_csv(ly,header=None)
        self.d1 = np.zeros((tem_dat.shape[0]*tem_dat.shape[1],1))
        self.d2 = np.zeros((self.disaster_number*2+1002,1))

        selected_factors = list(self.lisan_dict.keys())  # 返回类似于['name', 'age', 'birthday', 'sex']
        for fs in range(len(selected_factors)):
            if selected_factors[fs] in self.basic_fac:
                if selected_factors[fs] == 'altitude':
                    input_path = r"./data/re_altitude.tif "
                    print('高程编码')
                    self.d1,self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem((self.all_factor_name[0]))     #执行完毕后在下拉框增加选项
                    #self.ui.comboBox_2.addItem(self.modelname_list[1])
                if selected_factors[fs] == 'slope':
                    input_path = r"./data/reslope.tif "
                    print('坡度编码')
                    self.d1, self.d2= self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[1])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'aspect':

                    input_path = r"./data/re_aspecct.tif"
                    print('坡向编码')
                    self.d1, self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[2])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'profile curvature':
                    print('剖面曲率编码')
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_curve.tif "
                    self.d1,self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[3])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'plan curvature':
                    print('平面曲率编码')
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_plancure.tif"
                    self.d1,self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[4])

                if selected_factors[fs] == 'SDS':
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_sr.tif "
                    print('粗糙度编码')
                    self.d1,self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[5])

                if selected_factors[fs] == 'lithology':
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_yanzu.tif "
                    print('岩性编码')
                    self.d1,self.d2 = self.factor_fr(input_path,self.disaster_data,self.d1,self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    self.ui.comboBox.addItem(self.all_factor_name[6])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'rainfall':
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_rainfall.tif "
                    print('降雨量编码')
                    #JY = self.read_tif_file(input_path)[2]
                    self.d1,self.d2 = self.factor_fr(input_path, self.disaster_data, self.d1, self.d2)
                    #d3 = np.hstack((d3, dt1))
                    #d4 = np.hstack((d4, dt2))
                    #print("x=", d4.shape)
                    self.ui.comboBox.addItem(self.all_factor_name[7])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'NDVI':
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    input_path = r"./data/re_ndvi.tif "
                    print('NDVI编码')
                    self.d1,self.d2 = self.factor_fr(input_path, self.disaster_data, self.d1, self.d2)
                    #d3 = np.hstack((d3, np.array(dt1)))
                    #d4 = np.hstack((d4, np.array(dt2)))
                    self.ui.comboBox.addItem(self.all_factor_name[8])     #执行完毕后在下拉框增加选项

                if selected_factors[fs] == 'dis_to_river':
                    #函数应该包括 分级后的数据 作为输入  进行one_hot编码
                    # 执行函数————————
                    input_path = r"./data/re_dr.tif "
                    print('到河流距离编码')
                    self.d1,self.d2 = self.factor_fr(input_path, self.disaster_data, self.d1, self.d2)
                    #d3 = np.hstack((d3, np.array(dt1)))
                    #d4 = np.hstack((d4, np.array(dt2)))
                    self.ui.comboBox.addItem(self.all_factor_name[9])      #执行完毕后在下拉框增加选项
            else:
                input_path = r'./data/re_'+selected_factors[fs]+'.tif'
                print(selected_factors[fs]+'编码')
                self.d1, self.d2 = self.factor_fr(input_path, self.disaster_data, self.d1, self.d2)
                # d3 = np.hstack((d3, np.array(dt1)))
                # d4 = np.hstack((d4, np.array(dt2)))
                self.ui.comboBox.addItem(selected_factors[fs])  # 执行完毕后在下拉框增加选项

        self.d1 =self.d1[:,1:]   #预测数据
        self.d2 = self.d2[:,1:]  #训练数据
        #d3 = d3[:,1:]
        #d4 = d4[:,1:]
        print(self.d1.shape,self.d2.shape)

        pdd1 = pd.DataFrame(self.d1)
        pdd2 = pd.DataFrame(self.d2)
        pdd1.columns = selected_factors
        pdd2.columns = selected_factors
        print(pdd1.shape)

        biaoji = [0] * self.disaster_number + [1] * (self.disaster_number) + [0] * 1002
        y_dict = {}
        pdd2['y'] = biaoji
        self.y = pd.DataFrame(y_dict)

        pdd1.to_csv(r'.\data\xtest_dataset.csv',encoding='GBK',index=False)
        pdd2.to_csv(r'.\data\xtrain_dataset.csv',encoding='GBK',index=False)
        #np.savetxt(r'.\data\xtest_dataset.csv', self.d1, delimiter=',')
        #np.savetxt(r'.\data\xtrain_dataset.csv', self.d2, delimiter=',')

        #np.savetxt(r'.\data\xtrain_tree.csv', d4, delimiter=',')
        #np.savetxt(r'.\data\xtest_tree.csv', d3, delimiter=',')

    def checkVIF_new(self):
        sd2 = pd.read_csv(r'.\data\xtrain_dataset.csv', header=None)
        # self.d1 = np.array(sd1)
        # sd22 = self.reduce_mem_usage(sd2)
        selected_factors = list(self.lisan_dict.keys())
        fac = {}
        for i in range(len(selected_factors)):
            xx = {selected_factors[i]: sd2[:][i]}
            fac.update(xx)
            # print(fac)
        df = pd.DataFrame(fac)
        #df['c'] = 1
        df.insert(df.shape[1], 'c', 1)
        name = df.columns
        x = np.matrix(df)
        VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1])]
        VIF = pd.DataFrame({'feature':name,"VIF":VIF_list})
        #VIF = VIF.drop('c',axis = 0)
        del (VIF['c'])
        max_VIF = max(VIF_list)
        #print(VIF_list)

        for i in range(len(VIF['feature'])):
            self.ui.textBrowser_22.append(str(VIF['feature'][i])+': '+str(round(VIF['VIF'][i],3)))


    def show_zhuantitupian(self):#保存专题图片

        for i in self.all_factor_name:
            if self.ui.comboBox.currentText()==i:
                self.ui.lineEdit_15.setText(i)
                print(self.lisan_dict[i])
                self.ui.lineEdit_16.setText(str(self.lisan_dict[i]))
                path = self.factor_dic[i]
                data = self.read_tif_file(path)[2]
                ss_d = MinMaxScaler()
                ss_d = ss_d.fit(data)
                data1 = ss_d.transform(data)*255
                data2 = np.reshape(data1, (data1.shape[0], data1.shape[1], 1))
                #img = Image.fromarray(np.uint8(data2))
                plt.axis('off')
                plt.imshow(data2)
                plt.savefig(r'./results/aaa.jpg',dpi=600)
                #img.save('aaa.jpg')

                self.showImage = QPixmap(r'./results/aaa.jpg').scaled(self.ui.label_17.width(), self.ui.label_17.height())  # 适应窗口大小
                self.ui.label_17.setPixmap(self.showImage)  # 显示图片

    def save_zhuantitupian(self):#保存专题图片
        print( "提示选择路径，复制我文件")

        ##弹出对话框选择保存路径

    def xunlian_LJHG_model(self):
        LR = LogisticRegression()
        para_path = r'./data/配置文件.txt '
        Penalty_list = self.read_para(para_path, 'LR', 'Penalty')
        c_list = self.read_para(para_path, 'LR', 'c')
        Solver_list = self.read_para(para_path, 'LR', 'Solver')
        max_iter_list = self.read_para(para_path, 'LR', 'max_iter')
        para_dic = {'penalty': Penalty_list,
                    "C": c_list,
                    "solver": Solver_list,
                    "max_iter": max_iter_list,
                    }
        self.lrmodel,self.lrmodelpara=self.model_train(self.d2[:-1002],self.y[:-1002],LR,para_dic)
        joblib.dump(self.lrmodel, ".\model\LR_model.m")  # 保存模型

        self.ui.lineEdit_17.setText(str(self.lrmodelpara['penalty']))
        self.ui.lineEdit_18.setText(str(self.lrmodelpara["max_iter"]))
        self.ui.lineEdit_19.setText(str(round(self.lrmodelpara["C"],3)))
        self.ui.lineEdit_20.setText(str(self.lrmodelpara["solver"]))
        self.ui.comboBox_2.addItem(self.modelname_list[0])
        self.ui.comboBox_3.addItem(self.modelname_list[0])

    def xunlian_MLP_model(self):

        MLP = MLPClassifier(learning_rate_init=0.008,random_state=11)

        para_path = r'./data/配置文件.txt '
        #para_type = activation,solver,alpha
        #para_tyep = hidden_layer_sizes
        activation_list = self.read_para(para_path , 'MLP','activation')
        solver_list = self.read_para(para_path , 'MLP','solver')
        alpha_list = self.read_para(para_path, 'MLP', 'alpha')
        hidden_layer_sizes_tupple = [(24,8)]
        para_dic = {'solver': solver_list,
                      "activation": activation_list,
                      "hidden_layer_sizes": hidden_layer_sizes_tupple,
                      "alpha": alpha_list,
                      }
        self.dcgzjmodel,self.dcgzjmodelpara=self.model_train(self.d2[:-1002],self.y[:-1002],MLP,para_dic)
        joblib.dump(self.dcgzjmodel, ".\model\MLP_model.m")  #保存模型

        self.ui.lineEdit_21.setText(str(self.dcgzjmodelpara['hidden_layer_sizes']))
        self.ui.lineEdit_22.setText(str(self.dcgzjmodelpara["solver"]))
        self.ui.lineEdit_24.setText(str(self.dcgzjmodelpara["activation"]))
        self.ui.lineEdit_25.setText(str(round(self.dcgzjmodelpara["alpha"],3)))

        self.ui.comboBox_2.addItem(self.modelname_list[1])
        self.ui.comboBox_3.addItem(self.modelname_list[1])

    def xunlian_SVM_model(self):


        SVM = sklearn.svm.SVC(probability=True)

        para_path = r'./data/配置文件.txt '
        # para_type = activation,solver,alpha
        # para_tyep = hidden_layer_sizes
        c_list = self.read_para(para_path, 'SVM', 'c')
        kernel_list = self.read_para(para_path, 'SVM', 'kernel')
        gamma_list = self.read_para(para_path, 'SVM', 'gamma')
        total_list = self.read_para(para_path, 'SVM', 'tol')
        para_dic = {'C': c_list,
                    "kernel": kernel_list,
                    "gamma": gamma_list,
                    "tol": total_list,
                    }
        print(para_dic)
        self.svmmodel,self.svmmodelpara=self.model_train(self.d2[:-1002],self.y[:-1002], SVM,para_dic)
        joblib.dump(self.svmmodel, ".\model\SVM_model.m")  # 保存模型
        #round(roc, 3)
        self.ui.lineEdit_27.setText(str(round(self.svmmodelpara['C'],3)))
        self.ui.lineEdit_28.setText(str(self.svmmodelpara["kernel"]))
        self.ui.lineEdit_29.setText(str(round(self.svmmodelpara["gamma"],3)))
        self.ui.lineEdit_30.setText(str(round(self.svmmodelpara["tol"],3)))
        self.ui.comboBox_2.addItem(str(self.modelname_list[2]))
        self.ui.comboBox_3.addItem(str(self.modelname_list[2]))

    def xunlian_XGBOOST_model(self):
        XGBoost = xgboost.XGBClassifier(learning_rate =0.1,objective= 'binary:logistic')

        para_path = r'./data/配置文件.txt '
        # para_type = activation,solver,alpha
        # para_tyep = hidden_layer_sizes
        n_estimators_list = self.read_para(para_path, 'XGBOOST', 'n_estimators')
        max_depth_list = self.read_para(para_path, 'XGBOOST', 'max_depth')
        subsample_list = self.read_para(para_path, 'XGBOOST', 'subsample')
        colsample_bytree_list = self.read_para(para_path, 'XGBOOST', 'colsample_bytree')
        gamma_list = self.read_para(para_path, 'XGBOOST', 'gamma')
        min_child_weight_list = self.read_para(para_path, 'XGBOOST', 'min_child_weight')
        para_dic = {'n_estimators': n_estimators_list,
                    "max_depth": max_depth_list,
                    "subsample": subsample_list,
                    "colsample_bytree": colsample_bytree_list,
                    "gamma":gamma_list,
                    "min_child_weight":min_child_weight_list,
                    }

        self.xgboostmodel,self.xgboostmodelpara=self.model_train(self.d2[:-1002],self.y[:-1002], XGBoost , para_dic)
        joblib.dump(self.xgboostmodel, ".\model\XGBOOST_model.m")  # 保存模型

        self.ui.lineEdit_31.setText(str(round(self.xgboostmodelpara['subsample'],3)))
        self.ui.lineEdit_32.setText(str(round(self.xgboostmodelpara["colsample_bytree"],3)))
        self.ui.lineEdit_33.setText(str(self.xgboostmodelpara["gamma"]))
        self.ui.lineEdit_34.setText(str(self.xgboostmodelpara["min_child_weight"]))
        self.ui.lineEdit_35.setText(str(self.xgboostmodelpara["n_estimators"]))
        self.ui.lineEdit_36.setText(str(self.xgboostmodelpara["max_depth"]))
        self.ui.comboBox_2.addItem(self.modelname_list[3])
        self.ui.comboBox_3.addItem(self.modelname_list[3])

    def xunlian_RF_model(self):

        RFmodel = RandomForestClassifier(random_state=0)

        para_path = r'./data/配置文件.txt '
        # para_type = activation,solver,alpha
        # para_tyep = hidden_layer_sizes
        n_estimators_list = self.read_para(para_path, 'RandomForest', 'n_estimators')
        max_depth_list = self.read_para(para_path, 'RandomForest', 'max_depth')
        min_samples_leaf_list = self.read_para(para_path, 'RandomForest', 'min_samples_leaf')
        max_features_list = self.read_para(para_path, 'RandomForest', 'max_features')
        min_samples_split_list = self.read_para(para_path, 'RandomForest', 'min_samples_split')
        #min_child_weight_list = self.read_para(para_path, 'RandomForest', 'min_child_weight')
        para_dic = {'n_estimators': n_estimators_list,
                    "max_depth": max_depth_list,
                    "min_samples_leaf": min_samples_leaf_list,
                    "max_features": max_features_list,
                    "min_samples_split":min_samples_split_list,
                    }

        self.RFmodel,self.RFmodelpara=self.model_train(self.d2[:-1002],self.y[:-1002], RFmodel , para_dic)
        joblib.dump(self.RFmodel, ".\model\RF_model.m")  # 保存模型

        self.ui.lineEdit_49.setText(str(round(self.RFmodelpara['min_samples_leaf'],3)))
        self.ui.lineEdit_46.setText(str(round(self.RFmodelpara["max_features"],3)))
        self.ui.lineEdit_51.setText(str(round(self.RFmodelpara["min_samples_split"],3)))
        #self.ui.lineEdit_47.setText(str(self.xgboostmodelpara["min_child_weight"]))
        self.ui.lineEdit_50.setText(str(self.RFmodelpara["n_estimators"]))
        self.ui.lineEdit_48.setText(str(self.RFmodelpara["max_depth"]))
        self.ui.comboBox_2.addItem(self.modelname_list[4])
        self.ui.comboBox_3.addItem(self.modelname_list[4])


    def xunlian_C50_model1(self):
        #selected_factors = ['number', 'dem', 'rainfall', 'lulc', 'label']
        C50 = importr('C50')

        predictsk = """
            library(pacman)
            #install.packages('randomForest')
            library(randomForest)
            p_load(mlr,mlbench)
            data <- read.csv('./data/xtrain_dataset.csv')
            set.seed(5674)


            data$number = NULL # 人工去掉Id列
            #Mengubah tipe data
            #data$dem <- as.factor(data$dem)
            #data$rainfall <- as.factor(data$rainfall)
            #data$lulc <- as.factor(data$lulc)
            data$y <- as.factor(data$y)

            #Membagi data per kelas
            #vars <- c("dem", "lulc", "rainfall")
            #str(data[, c(vars, "label")])

            print(head(data))
            index <- sample(2,nrow(data),replace = TRUE,prob=c(0.7,0.3))

            traindata <- data[index==1,]

            testdata <- data[index==2,]

            #m <- randomForest(label~.,data=traindata,mtry=3,ntree=100, proximity=TRUE)
            m <- C5.0(y~.,data=traindata,trials = 3,proximity=TRUE)
            pred <- predict(m,newdata=testdata,type = "prob")
            print(pred)


            """
        r(predictsk)

        rescript = """
            library(pacman)
            #library(randomForest)
            p_load(mlr,mlbench)
            #install.packages('pROC')
            #install.packages('ggplot2')
            library(ggplot2)
            library("pROC")
            
            get = function(p1,p2,p3,p4){
                df <- read.csv('./data/xtrain_dataset.csv')
                set.seed(5674)

                #df$number = NULL # 人工去掉Id列

                df$y <- as.factor(df$y)
                tsk = makeClassifTask(data=df,target="y") # 建立任务
                #df$Id = NULL # 人工去掉Id列
                classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "y")

                #学习器任务
                classif.lrn = makeLearner("classif.C50", predict.type = "prob")
                #getParamSet(classif.lrn)

                #参数优化
                discrete_ps = makeParamSet(
                makeDiscreteParam("trials", values = seq(p1[1][1],p1[2][1],p1[3][1])),
                makeDiscreteParam("minCases",values = seq(p2[1][1],p2[2][1],p2[3][1])),
                makeDiscreteParam("CF",values = seq(p3[1][1],p3[2][1],p3[3][1])),
                makeDiscreteParam("sample",values = seq(p4[1][1],p4[2][1],p4[3][1]))
                )

                ctrl = makeTuneControlGrid()

                rdesc = makeResampleDesc("CV", iters = 5L)

                res = tuneParams("classif.C50", task = classif.task, resampling = rdesc,
                  par.set = discrete_ps, control = ctrl)

                rr = res$x
                para = c(rr$trials,rr$minCases,rr$CF,rr$sample)
                print(para)
                write.table(para, file='./model/para.csv',row.names=FALSE,sep=',')
                
                #lrn = setHyperPars(makeLearner("classif.C50"), rials = rr$rials, minCases = rr$minCases)

                #vars =c("dem", "lulc", "rainfall")
                #str(data[, c(vars, "label")])

                index <- sample(2,nrow(df),replace = TRUE,prob=c(0.7,0.3))

                traindata <- df[index==1,]
                testdata <- df[index==2,]

                #m <- C5.0(y~.,data = traindata,trials = rr$trials, minCases = rr$minCases, CF = rr$CF, sample = rr$sample,proximity=TRUE)
                m <- C5.0(y~.,data = traindata,trials = 31, minCases = 2, sample = 0.3,proximity=TRUE)
                
                pred <- predict(m,newdata=testdata,type = "prob")
                #print(pred)
                class <- predict(m,newdata=testdata)
                pred_train <- predict(m,newdata=traindata,type = "prob")
                class_train <- predict(m,newdata=traindata)
                #print(pred)
                
                #预测研究区的易发性
                #newdata2 = read.csv('./data/xtest_dataset.csv')
                #total_p = predict(m,newdata=newdata2,type = "prob")
                #write.table(pred, file='./results/total_C50.csv',row.names=FALSE,sep=',')
                
                
                #将真实值和预测值整合到一起
                obs_p_ran = data.frame(prob=class,obs=testdata$y)
                #输出混淆矩阵
                table = table(testdata$y,class,dnn=c("true","predict"))
                auc1 = sum(diag(table))/sum(table)
                
                
                write.table(pred, file='./data/predict_C50.csv',row.names=FALSE,sep=',')
                write.table(pred_train, file='./data/train_C50.csv',row.names=FALSE,sep=',')
                
                # 返回多个输出
                #a = auc1  #测试集的准确率
                #a_train = traindata$y
                #a_test = testdata$y
                #p_train = pred
                #p_test = pred
                aa = list(auc=auc1, a_train=traindata$y,a_test=testdata$y,p_train= pred_train, p_test = pred)
                return(aa)     
            }
            """

        #读取参数文件，转化为R语言C50能接收的格式
        para_path = './data/配置文件.txt '
        configuration = configparser.ConfigParser()
        configuration.read(para_path)

        def para(para_path, para_name):
            configuration.read(para_path)
            model_para = configuration.get('C5.0', para_name)
            para_list = model_para.split(",")
            if '.' in para_list[0]:
                para_list = [float(para_list[i]) for i in range(len(para_list))]
                return robjects.FloatVector(para_list)
            else:
                para_list = [int(para_list[i]) for i in range(len(para_list))]
                return robjects.IntVector(para_list)

        c50_p1 = para(para_path, 'trials')
        c50_p2 = para(para_path, 'minCases')
        c50_p3 = para(para_path, 'CF')
        c50_p4 = para(para_path, 'sample')
        print(c50_p1)
        #执行R语言中C50模型训练，参数优化
        rdf = r(rescript)
        ss = r.get(c50_p1,c50_p2,c50_p3,c50_p4)
        print('******************************')

        #保存模型参数
        self.c50_train = r['traindata']  #全局变量保存C50模型的训练集和测试集
        self.c50_test = r['testdata']
        para = pd.read_csv('./model/para.csv')   #交叉验证后参数优化结果
        p1,p2,p3,p4 = para['x'][0],para['x'][1],para['x'][2],para['x'][3]

        self.ui.lineEdit_44.setText(str(round(p1, 3)))  #trials,minCases,CF,sample
        self.ui.lineEdit_42.setText(str(round(p2, 3)))
        self.ui.lineEdit_43.setText(str(round(p3, 3)))
        self.ui.lineEdit_39.setText(str(round(p4, 3)))
        self.ui.comboBox_2.addItem(self.modelname_list[5])
        self.ui.comboBox_3.addItem(self.modelname_list[5])


        #保存准确率数据
        #测试集精度
        self.C50acc = list(ss.rx['auc'])[0][0]  # 测试集acc,准确率
        # print(auc)
        #print(ss)
        #print(list(ss.rx('a_test')))
        a_test = list(ss.rx('a_test')[0])
        #print(a_test)
        model_test = [int(a_test[i]-1) for i in range(len(a_test))]
        p_test = np.array(ss.rx('p_test').rx(1))
        #print(p_test)
        p_test = np.squeeze(p_test, 0)
        p_test = p_test[:, 1]
        #print(p_test)
        #p_test = [float(p_test[i]) for i in range(len(p_test))]
        print(model_test,p_test)
        self.C50fp_test, self.C50tp_test, threholds = metrics.roc_curve(model_test, p_test,pos_label=1)  # 用来画测试集roc 的两个参数
        self.C50test_auc = metrics.auc(self.C50fp_test, self.C50tp_test)  # 测试集auc
        print(self.C50test_auc)
        #训练集精度
        # ss(auc=auc1, a_train=traindata$label,a_test=testdata$label,p_train= pred_train, p_test = pred)
        a_train = list(ss.rx('a_train')[0])
        # print(a_test)
        model_train = [int(a_train[i]-1) for i in range(len(a_train))]
        p_train = np.array(ss.rx('p_train').rx(1))
        p_train = np.squeeze(p_train, 0)
        p_train = p_train[:, 1]
        self.C50fp_train, self.C50tp_train, threholds = metrics.roc_curve(model_train, p_train,pos_label=1)  # 用来画测试集roc 的两个参数
        self.C50train_auc = metrics.auc(self.C50fp_train, self.C50tp_train)  # 测试集auc

    def xunlian_C50_model(self):
        # selected_factors = ['number', 'dem', 'rainfall', 'lulc', 'label']
        C50 = importr('C50')

        predictsk = """
            library(pacman)
            #install.packages('randomForest')
            library(randomForest)
            p_load(mlr,mlbench)
            data <- read.csv('./data/xtrain_dataset.csv')
            set.seed(5674)


            data$number = NULL # 人工去掉Id列
            #Mengubah tipe data
            #data$dem <- as.factor(data$dem)
            #data$rainfall <- as.factor(data$rainfall)
            #data$lulc <- as.factor(data$lulc)
            data$y <- as.factor(data$y)

            #Membagi data per kelas
            #vars <- c("dem", "lulc", "rainfall")
            #str(data[, c(vars, "label")])

            print(head(data))
            index <- sample(2,nrow(data),replace = TRUE,prob=c(0.7,0.3))

            traindata <- data[index==1,]

            testdata <- data[index==2,]

            #m <- randomForest(label~.,data=traindata,mtry=3,ntree=100, proximity=TRUE)
            m <- C5.0(y~.,data=traindata,trials = 3,proximity=TRUE)
            pred <- predict(m,newdata=testdata,type = "prob")
            print(pred)


            """
        r(predictsk)

        re_para = """
            get_para = function(p1,p2,p3,p4){
                df <- read.csv('./data/xtrain_dataset.csv')
                set.seed(5674)

                #df$number = NULL # 人工去掉Id列

                df$y <- as.factor(df$y)
                tsk = makeClassifTask(data=df,target="y") # 建立任务
                #df$Id = NULL # 人工去掉Id列
                classif.task = makeClassifTask(id = "BreastCancer", data = df, target = "y")

                #学习器任务
                classif.lrn = makeLearner("classif.C50", predict.type = "prob")
                #getParamSet(classif.lrn)

                #参数优化
                discrete_ps = makeParamSet(
                makeDiscreteParam("trials", values = seq(p1[1][1],p1[2][1],p1[3][1])),
                makeDiscreteParam("minCases",values = seq(p2[1][1],p2[2][1],p2[3][1])),
                makeDiscreteParam("CF",values = seq(p3[1][1],p3[2][1],p3[3][1])),
                makeDiscreteParam("sample",values = seq(p4[1][1],p4[2][1],p4[3][1]))
                )

                ctrl = makeTuneControlGrid()

                rdesc = makeResampleDesc("CV", iters = 5L)

                res = tuneParams("classif.C50", task = classif.task, resampling = rdesc,
                  par.set = discrete_ps, control = ctrl)

                rr = res$x
                para = list(trials = rr$trials,minCases = rr$minCases,CF = rr$CF,sample = rr$sample)
                write.table(para, file='./model/para.csv',row.names=FALSE,sep=',')
                #return(para)
        }
        """
        rescript = """
            #library(pacman)
            #library(randomForest)
            #p_load(mlr,mlbench)
            #install.packages('pROC')
            #install.packages('ggplot2')
            #library(ggplot2)
            #library("pROC")

            get = function(p1,p2,p3,p4){
                df <- read.csv('./data/xtrain_dataset.csv')
                set.seed(5674)

                #df$number = NULL # 人工去掉Id列

                df$y <- as.factor(df$y)

                index <- sample(2,nrow(df),replace = TRUE,prob=c(0.7,0.3))

                traindata <- df[index==1,]
                testdata <- df[index==2,]

                #m <- C5.0(y~.,data = traindata,trials = rr$trials, minCases = rr$minCases, CF = rr$CF, sample = rr$sample,proximity=TRUE)
                m <- C5.0(y~.,data = traindata,trials = p1, minCases = p2, CF = p3, sample = p4,proximity=TRUE)

                pred <- predict(m,newdata=testdata,type = "prob")
                print(pred)
                class <- predict(m,newdata=testdata)
                pred_train <- predict(m,newdata=traindata,type = "prob")
                class_train <- predict(m,newdata=traindata)
                #print(pred)

                #预测研究区的易发性
                newdata2 = read.csv('./data/xtest_dataset.csv')
                print(nrow(newdata2))
                #newdata2$y <- as.factor(newdata2$y)
                total_p = predict(m,newdata=newdata2,type = "prob")
                write.table(total_p, file='./results/total_C50.csv',row.names=FALSE,sep=',')


                #将真实值和预测值整合到一起
                obs_p_ran = data.frame(prob=class,obs=testdata$y)
                #输出混淆矩阵
                table = table(testdata$y,class,dnn=c("true","predict"))
                auc1 = sum(diag(table))/sum(table)


                write.table(pred, file='./data/predict_C50.csv',row.names=FALSE,sep=',')
                write.table(pred_train, file='./data/train_C50.csv',row.names=FALSE,sep=',')

                # 返回多个输出
                #a = auc1  #测试集的准确率
                #a_train = traindata$y
                #a_test = testdata$y
                #p_train = pred
                #p_test = pred
                aa = list(auc=auc1, a_train=traindata$y,a_test=testdata$y,p_train= pred_train, p_test = pred)
                return(aa)     
            }
            """

        # 读取参数文件，转化为R语言C50能接收的格式
        para_path = './data/配置文件.txt '
        configuration = configparser.ConfigParser()
        configuration.read(para_path)

        def para(para_path, para_name):
            configuration.read(para_path)
            model_para = configuration.get('C5.0', para_name)
            para_list = model_para.split(",")
            if '.' in para_list[0]:
                para_list = [float(para_list[i]) for i in range(len(para_list))]
                return robjects.FloatVector(para_list)
            else:
                para_list = [int(para_list[i]) for i in range(len(para_list))]
                return robjects.IntVector(para_list)

        c50_p1 = para(para_path, 'trials')
        c50_p2 = para(para_path, 'minCases')
        c50_p3 = para(para_path, 'CF')
        c50_p4 = para(para_path, 'sample')
        # 执行R语言中C50模型训练，参数优化
        #r(re_para)
        #r.get_para(c50_p1,c50_p2,c50_p3, c50_p4)

        r(rescript)


        # 保存模型参数
        #self.c50_train = r['traindata']  # 全局变量保存C50模型的训练集和测试集
        #self.c50_test = r['testdata']
        para = pd.read_csv('./model/para.csv')  # 交叉验证后参数优化结果
        p1, p2, p3, p4 = para['trials'][0], para['minCases'][0], para['CF'][0], para['sample'][0]
        ss = r.get(p1,p2,p3,p4)
        print('******************************')
        self.ui.lineEdit_44.setText(str(round(p1, 3)))  # trials,minCases,CF,sample
        self.ui.lineEdit_42.setText(str(round(p2, 3)))
        self.ui.lineEdit_43.setText(str(round(p3, 3)))
        self.ui.lineEdit_39.setText(str(round(p4, 3)))
        self.ui.comboBox_2.addItem(self.modelname_list[5])
        self.ui.comboBox_3.addItem(self.modelname_list[5])

        # 保存准确率数据
        # 测试集精度
        self.C50acc = list(ss.rx['auc'])[0][0]  # 测试集acc,准确率
        # print(auc)
        # print(ss)
        # print(list(ss.rx('a_test')))
        a_test = list(ss.rx('a_test')[0])
        # print(a_test)
        model_test = [int(a_test[i] - 1) for i in range(len(a_test))]
        p_test = np.array(ss.rx('p_test').rx(1))
        # print(p_test)
        p_test = np.squeeze(p_test, 0)
        p_test = p_test[:, 1]
        # print(p_test)
        # p_test = [float(p_test[i]) for i in range(len(p_test))]
        #print(model_test, p_test)
        self.C50fp_test, self.C50tp_test, threholds = metrics.roc_curve(model_test, p_test,
                                                                        pos_label=1)  # 用来画测试集roc 的两个参数
        self.C50test_auc = metrics.auc(self.C50fp_test, self.C50tp_test)  # 测试集auc
        print(self.C50test_auc)
        # 训练集精度
        # ss(auc=auc1, a_train=traindata$label,a_test=testdata$label,p_train= pred_train, p_test = pred)
        a_train = list(ss.rx('a_train')[0])
        # print(a_test)
        model_train = [int(a_train[i] - 1) for i in range(len(a_train))]
        p_train = np.array(ss.rx('p_train').rx(1))
        p_train = np.squeeze(p_train, 0)
        p_train = p_train[:, 1]
        self.C50fp_train, self.C50tp_train, threholds = metrics.roc_curve(model_train, p_train,
                                                                          pos_label=1)  # 用来画测试集roc 的两个参数
        self.C50train_auc = metrics.auc(self.C50fp_train, self.C50tp_train)  # 测试集auc

    def all_model_train(self):

        sd2 = pd.read_csv(r'.\data\xtrain_dataset.csv')
        # self.d1 = np.array(sd1)
        #sd22 = self.reduce_mem_usage(sd2)
        del sd2['y']
        self.d2 = sd2
        """
        selected_factors = list(self.lisan_dict.keys())
        fac = {}
        for i in selected_factors:
            xx = {selected_factors[i]: sd2[i]}
            fac.update(xx)
            # print(fac)
        self.d2 = pd.DataFrame(fac)
        """
        #self.d2 = np.array(sd22)
        print(self.d2.shape)
        biaoji = [0] * self.disaster_number + [1] * (self.disaster_number)+[0]*1002
        y_dict = {}
        y_dict['y'] = biaoji
        self.y = pd.DataFrame(y_dict)

        if (self.ui.checkBox_9.isChecked()==True):
            self.modelpara[0]=1
            self.xunlian_LJHG_model()

        if (self.ui.checkBox_10.isChecked()==True):
            self.modelpara[1]=1
            self.xunlian_MLP_model()

        if (self.ui.checkBox_11.isChecked()==True):
            self.modelpara[2]=1
            self.xunlian_SVM_model()

        if (self.ui.checkBox_12.isChecked()==True):
            self.modelpara[3]=1
            self.xunlian_XGBOOST_model()
        if (self.ui.checkBox_15.isChecked()==True):
            self.modelpara[4]=1
            self.xunlian_RF_model()
        if (self.ui.checkBox_17.isChecked()==True):
            self.modelpara[5]=1
            self.xunlian_C50_model()

    def factor_importance(self):
        #self.xunlian_RF_model()
        #XXXX = r'.\model\XGBoost_model.m'
        #fa_model = joblib.load(XXXX)
        fa_model = xgboost.XGBClassifier(eval_metric='mlogloss').fit(self.d2, self.y)
        plt.figure(figsize=(6, 5), dpi=400)
        xgboost.plot_importance(fa_model, height=.5,
                                max_num_features=10,
                                grid=False,
                                show_values=True)
        plt.savefig('./results/factor_importance.jpg', dpi=400)
        #plt.show()

        self.showImageimpotance = QPixmap('./results/factor_importance.jpg').scaled(self.ui.label_66.width(),
                                                                self.ui.label_66.height())  # 适应窗口大小
        self.ui.label_66.setPixmap(self.showImageimpotance)  # 显示图片

    def read_tif_file(self,filepath):
        # 读取TIF文件
        # param filepath：文件路径
        dataset = gdal.Open(filepath)
        if dataset is None:
            print('The TIF file is invalid')
            sys.exit(1)
        projection = dataset.GetProjection()  # 地理投影坐标信息
        transform = dataset.GetGeoTransform()  # 仿射矩阵
        bands = dataset.RasterCount  # 波段数
        x = dataset.RasterXSize  # 东西方向矩阵长度，矩阵宽度
        y = dataset.RasterYSize  # 纬度方向矩阵长度，矩阵长度

        data_1 = np.zeros((x, y), dtype=np.float32)
        data_2 = np.zeros((x, y), dtype=np.float32)
        data_3 = np.zeros((x, y), dtype=np.float32)

        for i in range(bands):
            srcband = dataset.GetRasterBand(i + 1)
            if srcband is None:
                print('WARN: srcband is None: ' + str(i + 1) + filepath)
                continue
            if i + 1 == 1:
                arr = srcband.ReadAsArray(0, 0, x, y)
                data_1 = np.float32(arr)
            elif i + 1 == 2 and i + 1 <= bands:
                arr = srcband.ReadAsArray()
                data_2 = np.float32(arr)
            elif i + 1 == 3 and i + 1 <= bands:
                arr = srcband.ReadAsArray()
                data_3 = np.float32(arr)

        return projection, transform, data_1, data_2, data_3, x, y, bands

    # 写入栅格
    def preit_to_tif(self,data, input_filepath, output_filepath):

        projection, transform, data_1, data_2, data_3, x, y, bands = self.read_tif_file(input_filepath)
        if 'int8' in data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        driver = gdal.GetDriverByName('GTiff')
        print('---正在创建tif文件---')

        data1 = self.clip(data)

        out_tif = driver.Create(output_filepath, x, y, bands, datatype)
        out_tif.SetGeoTransform(transform)
        out_tif.SetProjection(projection)
        out_tif.GetRasterBand(1).WriteArray(data1)
        out_tif.FlushCache()  # 写入硬盘
        out_tif = None  # 释放数据
        #以dem作为基准，dem之外的全是nan，在dem之内的为实际数据。dem之内的nan由栅格分级的时候处理。


    # 读取CSV文件（经度、纬度）
    def read_csv_file(self,input_filepath):
        # 读取Excel表格信息

        disaster_data = pd.read_excel(input_filepath)  # 数据结构为DataFrame结构
        disaster_data_cut = disaster_data.loc[:, ['JD', 'WD']]
        disaster_data_array = disaster_data_cut.values
        return disaster_data_array  # numpy数组

    def clip(self,transformdata):
        slope1 = np.where(np.isnan(self.scope), np.nan, transformdata)  #self.scope 是高程图确定的面积，将这个作为基准
        return slope1

    # 栅格矩阵周边环绕一圈
    def Addround(self,data):
        # 在原栅格图像周围加一圈并返回
        # param  data:输入的np数组
        print('---正在运行self.Addround（）---')
        x, y = data.shape  # 数据行列数
        zbc = np.zeros((x + 2, y + 2))  # 创建加边界的数组
        zbc[1:-1, 1:-1] = data
        # 四边加格
        zbc[0, 1:-1] = data[0, :]
        zbc[-1, 1:-1] = data[-1, :]
        zbc[1:-1, 0] = data[:, 0]
        zbc[1:-1, -1] = data[:, -1]
        # 角点加格
        zbc[0, 0] = data[0, 0]
        zbc[0, -1] = data[0, -1]
        zbc[-1, 0] = data[-1, 0]
        zbc[-1, -1] = data[-1, -1]
        print('---self.Addround（）函数运行结束---')
        return zbc

    # 提取非灾点栅格
    def random_non_disaster_point(self,dem_path, disaster_number, disaster_data):
        size=30
        data11 = self.read_tif_file(dem_path)
        data1 = data11[2]
        zbc = self.Addround(data1)

        # 非灾点栅格随机选择函数

        coordinate = []
        random_coordinate = []
        zbc = np.where(zbc < -3000, -9999, zbc)

        # 筛选出非灾点栅格
        for i in range(zbc.shape[0]-2):
            for j in range(zbc.shape[1]-2):
                if [i,j] not in disaster_data and zbc[i,j]>-100:
                    coordinate.append([i, j])
        #print("coordinate=", len(coordinate))
        # 随机挑选出与灾点数量一致的非灾点栅格
        random.seed(10)
        n = random.sample(range(0, len(coordinate)), disaster_number+1002)
        for k in range(disaster_number+1002):
            # print(coordinate[100])
            random_coordinate.append(coordinate[n[k]])
            # n=n-1
        # random_coordinate.append([30,30])
        # print('********************')
        #print("random_coordinate=", random_coordinate)

        # 组成坐标列表数据集
        data_set = random_coordinate[:disaster_number]+disaster_data+random_coordinate[disaster_number:]
        return data_set  # List数据集

    # 导出灾点矩阵坐标编号
    def disaster_point(self,csv_filepath, tif_filepath):
        print('---正在运行self.disaster_point（）---')

        # 灾点文件
        self.disaster_point_info = self.read_csv_file(csv_filepath)  # 读取灾点文件

        # DEM文件
        read_tif = self.read_tif_file(tif_filepath)  # 读取DEM文件

        latitude = self.disaster_point_info[:, 0]  # 读取经度-X坐标
        longitude = self.disaster_point_info[:, 1]  # 读取维度-Y坐标

        disaster_number = self.disaster_point_info.shape[0]  # 计算灾点数量
        tif_transform = read_tif[1]
        x_star = tif_transform[0]  # 108.999861111112
        y_star = tif_transform[3]  # 34.5001388888888
        x_scale = tif_transform[1]
        y_scale = abs(tif_transform[5])
        x_gridnumber = read_tif[5]
        y_gridnumber = read_tif[6]
        x_toudian = []
        y_toudian = []

        # 灾点在矩阵数组中的投点
        for m in range(disaster_number):
            for i in range(x_gridnumber):
                if (x_star + i * x_scale) <= latitude[m] <= (x_star + (i + 1) * x_scale):
                    x_toudian.append(i)
                else:
                    continue
        disaster_toudian_1 = []
        for n in range(disaster_number):
            for j in range(y_gridnumber):
                if (y_star - (j + 1) * y_scale) <= longitude[n] <= (y_star - j * y_scale):
                    y_toudian.append(j)
                else:
                    continue
        for s in range(disaster_number):
            disaster_toudian_1.append([y_toudian[s], x_toudian[s]])

        return disaster_toudian_1, disaster_number

    def y(self,csv_filepath, tif_filepath):
        # tif_filepath = './data/landslide_point.csv'
        disaster_data1, disaster_number1 = self.disaster_point(csv_filepath, tif_filepath)
        tif_read1 = self.read_tif_file(tif_filepath)
        data1 = tif_read1[2]
        data_set = self.random_non_disaster_point(data1, disaster_number1, disaster_data1)

        return data_set



    # 读取高程

    # 计算高程,分区
    def Cac_altitude(self,input_filepath, scale):
        tif_read = self.read_tif_file(input_filepath)
        altitude = tif_read[2]
        altitude_copy = altitude.copy()
        for m in range(altitude.shape[0]):
            for n in range(altitude.shape[1]):
                if 0 < altitude[m][n] <= scale[len(scale) - 1]:
                    if altitude[m][n] <= scale[0]:
                        altitude[m][n] = 1
                    elif altitude[m][n] > scale[0]:
                        for i in range(len(scale) - 1):
                            if scale[i] < altitude[m][n] <= scale[i + 1]:
                                altitude[m][n] = i + 2
                elif scale[len(scale) - 1] <= altitude[m][n]:
                    altitude[m][n] = len(scale) + 1
                else:
                    altitude[m][n] = 0
        # altitude_csv = configuration.get("out_csv", "altitude_output_csv")
        # altitude_tif = configuration.get("out_tif", "altitude_output_tif")
        # np.savetxt(altitude_csv, altitude, delimiter=',')
        print("正在写入高程")
        altitude_tif = './data/re_altitude.tif'
        self.preit_to_tif(altitude, input_filepath, altitude_tif)
        return altitude, altitude_copy

    # 得到高程
    def Cal_height(self, input_filepath, size):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        altitude_tif = './data/altitude.tif'

        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）


    # 由高程导出坡度
    def Cal_slpoe(self,input_filepath, size):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        zbc = self.Addround(data)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / size / 2  # 一阶导
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / size / 2  # 一阶导

        # 计算坡度
        slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578
        slope_copy = slope.copy()
        slope_tif = './data/slope.tif'

        self.preit_to_tif(slope, input_filepath, slope_tif)  # 写文件（数据，输入数据路径，输出路径）

    # 高程分区离散化
    def Cac_slpoe(self,input_filepath, scale, size):  # （）
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        zbc = self.Addround(data)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / size / 2  # 一阶导
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / size / 2  # 一阶导

        # 计算坡度
        slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578
        slope_copy = slope.copy()

        # 坡度分级（需传入列表参数scale定义分级区间）
        for m in range(slope.shape[0]):
            for n in range(slope.shape[1]):
                if 0 < slope[m][n] <= scale[len(scale) - 1]:
                    if slope[m][n] <= scale[0]:
                        slope[m][n] = 1
                    elif slope[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < slope[m][n] <= scale[k + 1]:
                                slope[m][n] = k + 2
                elif scale[len(scale) - 1] <= slope[m][n] <= 90:
                    slope[m][n] = len(scale) + 1
                else:
                    slope[m][n] = 0

        re_slope_tif = './data/reslope.tif'
        print('正在写入坡度')
        self.preit_to_tif(slope, input_filepath, re_slope_tif)
        #return slope, slope_copy


    # 由高程文件计算坡向
    def cal_aspect(self,input_filepath, size):  # (高程文件路径，分辨率（默认30），保存高程文件的2路径)
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        print('---正在运行Cal_aspect（）---')
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        zbc = self.Addround(data)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / size / 2  # 一阶导
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / size / 2  # 一阶导

        # 计算坡向
        aspect = np.zeros([dx.shape[0], dx.shape[1]])
        for i in range(dx.shape[0]):
            for j in range(dx.shape[1]):
                x = dx[i, j]
                y = dy[i, j]
                aspect[i, j] = 180 + np.arctan(y / x) * 57.29578 - 90 * x / abs(x)
        aspect_tif = './data/aspect.tif'
        self.preit_to_tif(aspect, input_filepath, aspect_tif)

    # 坡向分级
    def Cac_Aspect(self,input_filepath):  # （）
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        zbc = self.Addround(data)
        #dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / size / 2  # 一阶导
        #dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / size / 2  # 一阶导
        aspect = np.zeros([data.shape[0], data.shape[1]])
        # 坡向分级
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                #x = dx[i, j]
                #y = dy[i, j]
                #aspect[i, j] = 270 + np.arctan(y / x) * 57.29578 - 90 * x / abs(x)
                if data[i, j] > 337.5 or data[i, j] <= 22.5:
                    aspect[i, j] = 1.
                elif 22.5 < data[i, j] <= 67.5:
                    aspect[i, j] = 2.
                elif 67.5 < data[i, j] <= 112.5:
                    aspect[i, j] = 3.
                elif 112.5 < data[i, j] <= 157.5:
                    aspect[i, j] = 4.
                elif 157.5 < data[i, j] <= 202.5:
                    aspect[i, j] = 5.
                elif 202.5 < data[i, j] <= 247.5:
                    aspect[i, j] = 6.
                elif 247.5 < data[i, j] <= 292.5:
                    aspect[i, j] = 7.
                elif 292.5 < data[i, j] <= 337.5:
                    aspect[i, j] = 8.
                else:
                    aspect[i, j] = -1
        print("正在写入坡向tif")
        re_aspect_tif = './data/re_aspecct.tif'
        self.preit_to_tif(aspect, input_filepath, re_aspect_tif)

        #return aspect  # numpy数组


    # 提取坐标周围8个栅格中像元值
    def D8(self,i, j, data):
        # 输出目标位置周围八个像元值
        # param  i：目标像元的X轴坐标
        # param  j：目标像元的Y轴坐标
        if i == 0:
            i = i + 1
        if j == 0:
            j = j + 1
        if i == data.shape[0] - 1:
            i = i - 1
        if j == data.shape[1] - 1:
            j = j - 1
        data_0 = data[i][j]
        data_1 = data[i][j + 1]
        data_2 = data[i + 1][j + 1]
        data_4 = data[i + 1][j]
        data_8 = data[i - 1][j - 1]
        data_16 = data[i][j - 1]
        data_32 = data[i - 1][j - 1]
        data_64 = data[i - 1][j]
        data_128 = data[i - 1][j + 1]
        a = [data_0, data_1, data_2, data_4, data_8, data_16, data_32, data_64, data_128]
        return a

    # 四次表面模型（曲率模型）
    def Quaternary_surface_model(self,i, j, data, L):
        # 四次表面模型计算剖面曲率值
        # param  i：目标像元的X轴坐标
        # param  j：目标像元的Y轴坐标
        # param  data: 输入的DEM矩阵数组
        # param  L：栅格单元分辨率
        A = self.D8(i, j, data)  # 输出坐标周围8个像元值
        Z1 = A[6]
        Z2 = A[7]
        Z3 = A[8]
        Z4 = A[5]
        Z5 = A[0]
        Z6 = A[1]
        Z7 = A[4]
        Z8 = A[3]
        Z9 = A[2]
        D = (((Z4 + Z6) / 2) - Z5) / (L ** 2)  #r
        E = (((Z2 + Z8) / 2) - Z5) / (L ** 2)  #t
        F = (-Z1 + Z3 + Z7 - Z9) / (4 * (L ** 2))  #s
        G = (-Z4 + Z6) / (2 * L)  #p
        H = (Z2 - Z8) / (2 * L)   #q
        curvature = ((-2*(D * (G ** 2) + E * (H ** 2) + F * G * H)) / (G ** 2 + H ** 2)) * (-100)  # 输出该坐标的剖面曲率值
        """if curvature > 0:
            curvature = 1
        elif curvature < 0:
            curvature = 1
        elif curvature < 0:
            curvature = -1
        else:
            curvature = 0"""
        #re_aspect_tif = './data/curvature.tif'
        #self.preit_to_tif(curvature, input_filepath, re_aspect_tif)
        return curvature

    # 计算剖面曲率
    def cal_curvature(self,input_filepath, L):
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        zbc = self.Addround(data)  # 四周加一圈环绕边界
        cur = np.zeros((zbc.shape[0], zbc.shape[1]))
        for i in range(zbc.shape[0]):
            for j in range(zbc.shape[1]):
                cur[i][j] = self.Quaternary_surface_model(i, j, zbc, L)
        #cur = np.where(abs(cur)>8, 0.1, cur)
        #cur = np.where(np.isnan(cur),0.1, cur)
        curve = cur[1:-1, 1:-1]

        curve_tif = './data/curve.tif'
        self.preit_to_tif(curve, input_filepath, curve_tif)
        #return curve

    def cal_cracurve(self,input_filepath,L):
        self.cal_aspect(input_filepath, L)
        aspect_path = r'.\data\slope.tif'

        tif_read = self.read_tif_file(aspect_path)
        data = tif_read[2]
        zbc = self.Addround(data)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / L / 2  # 一阶导
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / L / 2  # 一阶导

        # 计算坡度
        plancurve = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578

        plancurve_path = r'.\data\curve.tif'
        self.preit_to_tif(plancurve, input_filepath, plancurve_path)  # 写文件（数据，输入数据路径，输出路径）

    def cal_plancurve(self,input_filepath,L):
        self.cal_aspect(input_filepath, L)
        aspect_path = r'.\data\aspect.tif'

        tif_read = self.read_tif_file(aspect_path)
        data = tif_read[2]
        zbc = self.Addround(data)
        dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / L / 2  # 一阶导
        dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / L / 2  # 一阶导

        # 计算坡度
        plancurve = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578

        plancurve_path = r'.\data\plancurve.tif'
        self.preit_to_tif(plancurve, input_filepath, plancurve_path )  # 写文件（数据，输入数据路径，输出路径）

    def SDS(self,input_filepath,L):

        self.Cal_slpoe(input_filepath, L)
        slope_path = r'.\data\slope.tif'
        tif_read = self.read_tif_file(slope_path)
        data = tif_read[2]
        sds = 1/np.cos(data*3.1415926/180)  #sds = 1/cos(s*3.1415926/180)  s is slope
        sds = np.where(abs(sds) > 2, 1.049, sds)

        SDS_path = r'.\data\surface roughness.tif'
        self.preit_to_tif(sds, input_filepath, SDS_path)  # 写文件（数据，输入数据路径，输出路径）

    def factor_level(self,input_path,scale,re_factor_name):
        tif_read = self.read_tif_file(input_path)
        data = tif_read[2]
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                if data[m][n] <= scale[len(scale) - 1]:
                    if data[m][n] <= scale[0]:
                        data[m][n] = 1
                    elif data[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < data[m][n] <= scale[k + 1]:
                                data[m][n] = k + 2
                elif scale[len(scale) - 1] <= data[m][n] :
                    data[m][n] = len(scale) + 1
                else:
                    data[m][n] = 0



        #re_slope_tif = './data/reslope.tif'
        print('正在写入坡度')
        self.preit_to_tif(data, input_path, re_factor_name)

    def distence_to_river(self,input_path,scale,re_factor_name):
        tif_read = self.read_tif_file(input_path)
        data = tif_read[2]
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                if -100 < data[m][n] <= scale[len(scale) - 1]:
                    if data[m][n] <= scale[0]:
                        data[m][n] = 1
                    elif data[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < data[m][n] <= scale[k + 1]:
                                data[m][n] = k + 2
                elif scale[len(scale) - 1] <= data[m][n]:
                    data[m][n] = len(scale) + 1
                else:
                    data[m][n] = 0



        #re_slope_tif = './data/reslope.tif'
        print('正在写入距河流距离')
        self.preit_to_tif(data, input_path, re_factor_name)

    #剖面曲率分级
    def Cac_curvature(self,input_filepath,scale):
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        #zbc = self.Addround(data)  # 四周加一圈环绕边界
        cur = np.zeros((data.shape[0], data.shape[1]))
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                if -50 <= data[m][n] <= scale[len(scale) - 1]:
                    if data[m][n] <= scale[0]:
                        cur[m][n] = 1
                    elif data[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < data[m][n] <= scale[k + 1]:
                                cur[m][n] = k + 2
                elif scale[len(scale) - 1] <= data[m][n] <= 50:
                    cur[m][n] = len(scale) + 1
                else:
                    cur[m][n] = 0
                #cur[m][n] = data[m][n]
        curve = cur[1:-1, 1:-1]
        print('正在写入剖面曲率')
        curve_tif = './data/re_curve.tif'
        self.preit_to_tif(cur, input_filepath, curve_tif)


    # 一般因子分级，使用范围-10，100
    def divise_factor(self, input_filepath, scale, out_path):
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        # zbc = self.Addround(data)  # 四周加一圈环绕边界
        cur = np.zeros((data.shape[0], data.shape[1]))
        for m in range(data.shape[0]):
            for n in range(data.shape[1]):
                if data[m][n] <= scale[len(scale) - 1]:
                    if data[m][n] <= scale[0]:
                        cur[m][n] = 1
                    elif data[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < data[m][n] <= scale[k + 1]:
                                cur[m][n] = k + 2
                elif scale[len(scale) - 1] <= data[m][n]:
                    cur[m][n] = len(scale) + 1
                else:
                    cur[m][n] = 0
                    # cur[m][n] = data[m][n]
        curve = cur[1:-1, 1:-1]
        print('正在进行因子分级')
        #curve_tif = './data/re_curve.tif'
        self.preit_to_tif(cur, input_filepath, out_path)

    # 读取地层岩性tif,不用划分等级
    def Cac_yanzu(self,input_filepath):
        tif_read = self.read_tif_file(input_filepath)
        yanzu = tif_read[2]
        return yanzu

    # 计算灾点密度，这个因子先不写
    def Cac_density(self,data, disaster_data, size):
        # param: input_file  输入的底图文件，用以获取灾点投影的矩阵代号值
        y = data.shape[1]
        x = data.shape[0]
        density = np.zeros([x, y])


        for i in range(len(disaster_data)):  # 遍历所有的灾点
            count = 0
            list_1 = self.circle_search(i=disaster_data[i][0], j=disaster_data[i][1], data=data,
                                   size=size)  # 遍历所有灾点一定范围内的栅格坐标
            for j in range(len(list_1)):
                if list_1[j] in disaster_data:
                    count = count + 1
            for m in range(len(list_1)):
                if density[list_1[m][0]][list_1[m][1]] < count:
                    density[list_1[m][0]][list_1[m][1]] = count
                elif density[list_1[m][0]][list_1[m][1]] == count:
                    density[list_1[m][0]][list_1[m][1]] = count + 1
        for k in range(x):
            for n in range(y):
                if density[k][n] <= 0:
                    density[k][n] = 0
                elif 0 < density[k][n] <= 2:
                    density[k][n] = 1
                elif 2 < density[k][n] <= 4:
                    density[k][n] = 2
                elif 4 < density[k][n] <= 7:
                    density[k][n] = 3
                else:
                    density[k][n] = 4
        return density

    # 计算植被归一化指数NDVI
    def Cac_NDVI(self,input_filepath, scale):
        tif_read = self.read_tif_file(input_filepath)
        ndvi = tif_read[2]
        for m in range(ndvi.shape[0]):
            for n in range(ndvi.shape[1]):
                if -1 <= ndvi[m][n] < scale[len(scale) - 1]:
                    if -1 < ndvi[m][n] <= scale[0]:
                        ndvi[m][n] = 1
                    elif ndvi[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < ndvi[m][n] <= scale[k + 1]:
                                ndvi[m][n] = k + 2
                elif scale[len(scale) - 1] <= ndvi[m][n]:
                    ndvi[m][n] = len(scale) + 1
                else:
                    ndvi[m][n] = 0
        altitude_tif = './data/re_ndvi.tif'
        self.preit_to_tif(ndvi, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）
        #return ndvi
    #读取fit写入tif
    def write_rainfall(self, input_filepath):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        data = np.where(np.isnan(data),0,data)
        data = np.where(data<0,0,data)
        altitude_tif = './data/rainfall.tif'
        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）

    def write_ndvi(self, input_filepath):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        altitude_tif = './data/ndvi.tif'
        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）

    def write_yanzu(self, input_filepath):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        altitude_tif = './data/yanzu.tif'
        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）

    def write_yanzu2(self, input_filepath):
        # param  dx:  x轴梯度
        # param  dy： y轴梯度
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        data = np.where(np.isnan(data),0,data)
        altitude_tif = './data/re_yanzu.tif'
        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）

    def write_river(self,input_filepath):
        tif_read = self.read_tif_file(input_filepath)
        data = tif_read[2]
        altitude_tif = './data/dis_river.tif'
        self.preit_to_tif(data, input_filepath, altitude_tif)  # 写文件（数据，输入数据路径，输出路径）

    # 计算降雨量分布
    def Cac_Rainfall_distribution(self,input_filepath, scale):
        tif_read = self.read_tif_file(input_filepath)
        rainfall = tif_read[2]
        for m in range(rainfall.shape[0]):
            for n in range(rainfall.shape[1]):
                if -1 <= rainfall[m][n] < scale[len(scale) - 1]:
                    if -1 < rainfall[m][n] <= scale[0]:
                        rainfall[m][n] = 1
                    elif rainfall[m][n] > scale[0]:
                        for k in range(len(scale) - 1):
                            if scale[k] < rainfall[m][n] <= scale[k + 1]:
                                rainfall[m][n] = k + 2
                elif scale[len(scale) - 1] <= rainfall[m][n]:
                    rainfall[m][n] = len(scale) + 1
                else:
                    rainfall[m][n] = 0
        re_rainfall = './data/re_rainfall.tif'
        self.preit_to_tif(rainfall, input_filepath, re_rainfall)
        #return rainfall

    # 坐标范围搜索
    def circle_search(self,i, j, data, size):
        list_1 = []
        y = data.shape[1]
        x = data.shape[0]
        for m in range(100):
            for n in range(100):
                if math.sqrt((m * size) ** 2 + (n * size) ** 2) < 1000:
                    if m == 0 and n == 0:
                        list_1.append([i, j])
                    if m == 0 and n != 0:
                        list_1.append([i, j + n])
                        list_1.append([i, j - n])
                    if n == 0 and m != 0:
                        list_1.append([i + m, j])
                        list_1.append([i - m, j])
                    if i + m <= x and j + n <= y and m != 0 and n != 0:
                        list_1.append([i + m, j + n])
                    if i + m <= x and j - n > 0 and m != 0 and n != 0:
                        list_1.append([i + m, j - n])
                    if i - m > 0 and j + n <= y and m != 0 and n != 0:
                        list_1.append([i - m, j + n])
                    if i - m > 0 and j - n > 0 and m != 0 and n != 0:
                        list_1.append([i - m, j - n])
        return list_1

#########因子编码
    def factor_code(self,input_path,d1,d2):
        tif_read = self.read_tif_file(input_path)
        data_set = tif_read[2]

        data_set = np.where(np.isnan(data_set),0,data_set)

        # 生成预测数据集
        data_pre = []
        print("开始测试集编码")
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]):
                data_pre.append(data_set[i][j])
        altitude_pre_dict = {'altitude': data_pre}
        altitude_pre_frame = pd.DataFrame(altitude_pre_dict)
        altitude_pre_values = altitude_pre_frame.values
        onehot = OneHotEncoder(categories='auto', sparse=False, handle_unknown='error')
        al = onehot.fit(altitude_pre_values)
        pre_x1 = al.transform(altitude_pre_values)
        print(pre_x1.shape)
        d1 = np.hstack((d1,pre_x1))
        #print('d1 = ',d1.shape)
        altitude_pre_dict,altitude_pre_frame,pre_x1,data_pre = [],[],[],[]


        #生成训练集
        print('开始训练集编码')
        data_set_altitude = []
        for i in range(self.disaster_number):
            data_set_altitude.append(data_set[self.data_y[i * 2][0], self.data_y[i * 2][1]])
            data_set_altitude.append(data_set[self.data_y[2 * i + 1][0], self.data_y[2 * i + 1][1]])
        for i in range(self.disaster_number,self.disaster_number+1002):
            data_set_altitude.append(data_set[self.data_y[i + 1][0], self.data_y[i + 1][1]])
            # print([landslide_set[2*i+1][0],landslide_set[2*i+1][1]])
        altitude_dict = {'altitude': data_set_altitude}
        altitude_frame = pd.DataFrame(altitude_dict)
        altitude_values = altitude_frame.values
        # onehot = OneHotEncoder(categories='auto', sparse=False, handle_unknown='error')
        x = al.transform(altitude_values)
        d2 = np.hstack((d2,x))
        #print("x=", d2.shape)
        #print("2222")
        data_set_altitude,altitude_dict,altitude_frame,x = [],[],[],[]
        return d1,d2,altitude_pre_values, altitude_values  #测试集，训练集

    def factor_fr(self,input_path, landslide_position,d1,d2):
        tif_read = self.read_tif_file(input_path)
        data_set = tif_read[2]

        # 统计各分级因子的数量
        data_pre = []
        landslide_factor = []
        print("开始测试集编码")
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]):
                data_pre.append(data_set[i][j])
                if [i, j] in landslide_position:
                    landslide_factor.append(data_set[i, j])

        data1 = np.where(np.isnan(data_pre), -9999, data_pre)
        ar, num = np.unique(data1, return_counts=True)

        com = np.array([ar, num])
        # print(com)

        for i in range(len(ar)):
            if ar[i] < 0:
                com = np.delete(com, [i], axis=1)  # 删除某列,[ni,si]
                # print('*********')

        landslide_factor1 = np.where(np.isnan(landslide_factor), -9999, landslide_factor)
        lar, lnum = np.unique(landslide_factor1, return_counts=True)
        lcom = np.array([lar, lnum])
        for i in range(len(lar)):
            if lar[i] < 0:
                lcom = np.delete(lcom, [i], axis=1)  # 删除某列,[ni,si]
                # print('*********')

        # 计算频率比
        dataset1 = []
        for i in range(len(com[0, :])):
            s1 = 0
            for j in range(len(lcom[0, :])):
                if com[0, i] == lcom[0, j]:
                    s1 = j
            dataset1.append([com[0, i], com[1, i], lcom[1, s1]])

        dataset2 = np.array(dataset1)
        print(dataset2)
        level = dataset2[:, 0]
        total_area = np.sum(dataset2[:, 1])
        total_landslide = np.sum(dataset2[:, 2])
        # print(total_area,total_landslide)
        fr_level = np.zeros((len(level), 2))
        for i in range(len(level)):
            # print(dataset2[i,2])
            fr = (dataset2[i, 2] / total_landslide) / (dataset2[i, 1] / total_area)
            fr_level[i, :] = [level[i], fr]


        # 研究区分级区域赋值--fr-value
        for i in range(len(fr_level)):
            data_set = np.where(data_set == fr_level[i, 0], fr_level[i, 1], data_set)

        # 训练集数据
        data_set_altitude = []
        for i in range(self.disaster_number):
            data_set_altitude.append(data_set[self.data_y[i * 2][0], self.data_y[i * 2][1]])
            data_set_altitude.append(data_set[self.data_y[2 * i + 1][0], self.data_y[2 * i + 1][1]])
        for i in range(self.disaster_number*2, self.disaster_number*2 + 1002):
            data_set_altitude.append(data_set[self.data_y[i-1][0], self.data_y[i-1][1]])
        #al = StandardScaler()
        #pre_x1 = al.fit_transform(np.array(data_set_altitude))
        temd = np.array(data_set_altitude)
        temd = np.reshape(temd,(-1,1))
        print(temd.shape)
        d2 = np.hstack((d2, temd))

        #测试集数据，整个研究区域
        #x = al.transform(np.array(data_set))
        x = np.array(data_set)
        temx = x.reshape((data_set.shape[0]*data_set.shape[1],1))
        #print(temx.shape)
        d1 = np.hstack((d1, temx))   #ceshiji

        return d1,d2

    def factor_tr(self,input_path,landslide_position):
        tif_read = self.read_tif_file(input_path)
        data_set = tif_read[2]

        # 生成预测数据集
        data_pre = []
        print("开始测试集编码")
        for i in range(data_set.shape[0]):
            for j in range(data_set.shape[1]):
                data_pre.append(data_set[i][j])
        altitude_pre_dict = {'altitude': data_pre}
        altitude_pre_frame = pd.DataFrame(altitude_pre_dict)
        altitude_pre_values = altitude_pre_frame.values
        onehot = OneHotEncoder(categories='auto', sparse=False, handle_unknown='error')
        al = onehot.fit(altitude_pre_values)
        pre_x1 = al.transform(altitude_pre_values)
        print(pre_x1.shape)
        d1 = np.hstack((d1, pre_x1))
        # print('d1 = ',d1.shape)
        altitude_pre_dict, altitude_pre_frame, pre_x1, data_pre = [], [], [], []

        # 生成训练集
        print('开始训练集编码')
        data_set_altitude = []
        for i in range(self.disaster_number):
            data_set_altitude.append(data_set[self.data_y[i * 2][0], self.data_y[i * 2][1]])
            data_set_altitude.append(data_set[self.data_y[2 * i + 1][0], self.data_y[2 * i + 1][1]])
            # print([landslide_set[2*i+1][0],landslide_set[2*i+1][1]])
        altitude_dict = {'altitude': data_set_altitude}
        altitude_frame = pd.DataFrame(altitude_dict)
        altitude_values = altitude_frame.values
        # onehot = OneHotEncoder(categories='auto', sparse=False, handle_unknown='error')
        x = al.transform(altitude_values)
        d2 = np.hstack((d2, x))
        # print("x=", d2.shape)
        # print("2222")
        data_set_altitude, altitude_dict, altitude_frame, x = [], [], [], []
        return d1, d2, altitude_pre_values, altitude_values  # 测试集，训练集


    def reduce_mem_usage(self,df):
        """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
        """
        start_mem = df.memory_usage().sum()
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum()
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        return df



    def model_train(self,x,y,model,para_dict):
        ss_x = StandardScaler()
        x1 = ss_x.fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.3, random_state=13)
        #y_train = y_train.ravel()
        #y_test = y_test.ravel()

        estimator = GridSearchCV(model, param_grid=para_dict, cv=5)
        estimator.fit(x_train, y_train)

        # 方法1：计算准确率
        #MLP_score = estimator.score(x_test, y_test)
        # 最佳参数：best_params_
        print("最佳参数：\n", estimator.best_params_)

        return estimator,estimator.best_params_


    def model_use_showroc(self):
        cur_text = self.ui.comboBox_2.currentText()  # 得到复选框的选中的变量

        if cur_text ==self.modelname_list[0]:#ljhg path
            XXXX = r'.\model\LR_model.m'
            roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(roc,3)))
            self.ui.lineEdit_38.setText(str(round(acc,3)))
            self.ui.lineEdit_40.setText(str(round(yt_auc, 3)))  # 训练集AUC
        if cur_text == self.modelname_list[1]:
            XXXX = r'.\model\MLP_model.m'
            roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(roc,3)))
            self.ui.lineEdit_38.setText(str(round(acc,3)))
            self.ui.lineEdit_40.setText(str(round(yt_auc, 3)))  # 训练集AUC
        if cur_text == self.modelname_list[2]:
            XXXX = r'.\model\SVM_model.m'
            roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(roc,3)))
            self.ui.lineEdit_38.setText(str(round(acc,3)))
            self.ui.lineEdit_40.setText(str(round(yt_auc, 3)))  # 训练集AUC
        if cur_text == self.modelname_list[3]:
            XXXX = r'.\model\XGBOOST_model.m'
            roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(roc,3)))
            self.ui.lineEdit_38.setText(str(round(acc,3)))   #测试集AUC
            self.ui.lineEdit_40.setText(str(round(yt_auc, 3)))  #训练集AUC

        if cur_text == self.modelname_list[4]:
            XXXX = r'.\model\RF_model.m'
            roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(roc,3)))
            self.ui.lineEdit_38.setText(str(round(acc,3)))   #测试集AUC
            self.ui.lineEdit_40.setText(str(round(yt_auc, 3)))  #训练集AUC

        if cur_text == self.modelname_list[5]:
            #XXXX = r'.\model\C5.0_model.m'
            #roc,acc,yt_auc=self.calculat_roc_acc(XXXX,cur_text)
            self.ui.lineEdit_37.setText(str(round(self.C50acc,3)))
            self.ui.lineEdit_38.setText(str(round(self.C50test_auc,3)))   #测试集AUC
            self.ui.lineEdit_40.setText(str(round(self.C50train_auc, 3)))  #训练集AUC

            plt.figure(figsize=(7.8, 6), dpi=400)
            plt.plot(self.C50fp_train, self.C50tp_train, color='red', lw=2, label='训练集AUC = %0.3f' % self.C50train_auc)
            plt.plot(self.C50fp_test, self.C50tp_test, color='darkorange', lw=2, label='测试集AUC = %0.3f' % self.C50test_auc)

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([-0.01, 1.0])
            plt.ylim([0.0, 1.01])
            plt.legend(loc="lower right", prop={'size': 20})
            plt.xlabel('假正率', fontsize=20)
            plt.ylabel('真正率', fontsize=20)
            plt.xticks(size=18)
            plt.yticks(size=18)
            # plt.legend()
            TK = plt.gca()
            TK.spines['bottom'].set_linewidth(1.6)  # 图框下边
            TK.spines['left'].set_linewidth(1.6)  # 图框左边
            TK.spines['top'].set_linewidth(1.6)  # 图框上边
            TK.spines['right'].set_linewidth(1.6)
            plt.savefig('./results/ROC.jpg', dpi=400)



        self.showImageroc = QPixmap('./results/ROC.jpg').scaled(self.ui.label_42.width(), self.ui.label_42.height())  # 适应窗口大小
        self.ui.label_42.setPixmap(self.showImageroc)  # 显示图片

    def yifaxing_pre(self):

        cur_text = self.ui.comboBox_3.currentText()  # 得到复选框的选中的变量
        #cur_text = self.modelname_list
        zone_lure = [0.2,0.4,0.6,0.85,1]
        dem_path = r'.\data\altitude.tif'
        configuration = configparser.ConfigParser()
        configuration.read(r'.\data\配置文件.txt')
        zone_para = configuration.get('zone_ruler', 'z')
        #para_list = zone_para .split(",")

        #if cur_text == self.modelname_list[0] or cur_text == self.modelname_list[1] or cur_text == self.modelname_list[2]:
            #dat1 = pd.read_csv(r'.\data\xtest_dataset.csv', header=None).fillna(0)
            # dat2 = pd.read_csv(r'.\data\xtrain_dataset.csv', header=None).fillna(0)
            #sd1 = self.reduce_mem_usage(dat1)


            #zone_lure = [float(para_list[i]) for i in range(len(para_list))]
        if cur_text ==self.modelname_list[0]:#ljhg path
            model = r'.\model\LR_model.m'  #输入路径
            landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            data1 = np.array(landslide_pro)
            save_path = r'./results/LR_landslide_pro.tif'
            self.preit_to_tif(data1,dem_path,save_path)
            """self.ui.lineEdit_39.setText(str(one))
            self.ui.lineEdit_40.setText(str(two))
            self.ui.lineEdit_41.setText(str(thr))
            self.ui.lineEdit_43.setText(str(fou))"""
        if cur_text == self.modelname_list[1]:
            model = r'.\model\MLP_model.m'  # 输入路径
            landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            data1 = np.array(landslide_pro)
            save_path = r'./results/MLP_landslide_pro.tif'
            self.preit_to_tif(data1, dem_path, save_path)
            """self.ui.lineEdit_39.setText(str(one))
            self.ui.lineEdit_40.setText(str(two))
            self.ui.lineEdit_41.setText(str(thr))
            self.ui.lineEdit_43.setText(str(fou))"""
        if cur_text == self.modelname_list[2]:
            model = r'.\model\SVM_model.m'  # 输入路径
            landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            data1 = np.array(landslide_pro)
            save_path = './results/SVM_landslide_pro.tif'
            self.preit_to_tif(data1, dem_path, save_path)
            """self.ui.lineEdit_39.setText(str(one))
            self.ui.lineEdit_40.setText(str(two))
            self.ui.lineEdit_41.setText(str(thr))
            self.ui.lineEdit_43.setText(str(fou))"""
        if cur_text == self.modelname_list[3]:

            model = r'.\model\XGBOOST_model.m'  # 输入路径
            landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            data1 = np.array(landslide_pro)
            save_path = './results/XGBOOST_landslide_pro.tif'
            self.preit_to_tif(data1, dem_path, save_path)
            """self.ui.lineEdit_39.setText(str(one))
            self.ui.lineEdit_40.setText(str(two))
            self.ui.lineEdit_41.setText(str(thr))
            self.ui.lineEdit_43.setText(str(fou))"""
        if cur_text == self.modelname_list[4]:

            model = r'.\model\RF_model.m'  # 输入路径
            landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            data1 = np.array(landslide_pro)
            save_path = './results/RF_landslide_pro.tif'
            self.preit_to_tif(data1, dem_path, save_path)

        if cur_text == self.modelname_list[5]:

            #model = r'.\model\RF_model.m'  # 输入路径
            #landslide_pro,one,two,thr,fou = self.zone(model,zone_lure)
            path = r'.\data\altitude.tif'
            dem = self.read_tif_file(path)[2]
            dat1 = pd.read_csv(r'.\results\total_C50.csv')
            prop = dat1['1']
            landslide_pro = np.array(prop).reshape((dem.shape[0], dem.shape[1]))
            data1 = np.array(landslide_pro)
            save_path = './results/C50_landslide_pro.tif'
            self.preit_to_tif(data1, dem_path, save_path)

            # 绘制易发性图
            tras_POF = (1 - landslide_pro) * 255
            data3 = np.where(np.isnan(self.scope), np.nan, tras_POF)
            data2 = np.reshape(data3, (tras_POF.shape[0], tras_POF.shape[1], 1))

            # img = Image.fromarray(np.uint8(data2))
            plt.figure(figsize=(6, 6), dpi=600)
            plt.axis('off')
            plt.imshow(data2, cmap='Spectral')
            # plt.axis('off')
            plt.savefig('./results/' + 'pof_zone.jpg', dpi=600)


        self.showImagefenqu = QPixmap(r'./results/pof_zone.jpg').scaled(self.ui.label_54.width(), self.ui.label_54.height())  # 适应窗口大小
        self.ui.label_54.setPixmap(self.showImagefenqu)  # 显示图片

        self.showImagefenqu = QPixmap(r'./results/pro_label.jpg').scaled(self.ui.label_141.width(),self.ui.label_141.height())  # 适应窗口大小
        self.ui.label_141.setPixmap(self.showImagefenqu)  # 显示图片


    def calculat_roc_acc(self,path,model_name):
        model = joblib.load(path)

        ss_x = StandardScaler()
        x1 = ss_x.fit_transform(self.d2)
        #这个划分是用来计算AUC值的，下面一个的划分是用来绘制ROC曲线的，两者之间有一点微小的区别
        x_train, x_test, y_train, y_test = train_test_split(x1[:-1002], self.y[:-1002], test_size=0.3, random_state=128)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(x1[-210:-10], self.y[-210:-10], test_size=0.7, random_state=135)
        #x_train1 =
        #if "XGBOOST" in path:
            #x_tr1 = self.reduce_mem_usage(pd.read_csv(r'./data/xtrain_tree.csv',header=None))
            #y = model.model.predict_proba(x_xgboost)
        #else:
        #x_tr1 = self.reduce_mem_usage(pd.read_csv('./data/xtrain_dataset.csv',header=None))

        #测试集误差
        x_tr = np.array(x_test)
        y_p = model.predict_proba(x_tr)
        model_pred = y_p[:, 1]
        y_plable = model.predict(x_tr)

        model_acc = metrics.accuracy_score(y_test,y_plable)
        # 计算ROC,AUC
        fpr,tpr,threholds = metrics.roc_curve(y_test,model_pred)
        #model_auc = metrics.auc(fpr,tpr)



        #画图数据得到
        x_tr2 = np.array(x_test2)
        x_tr1 = np.vstack((x_tr,x_tr2))
        y_test1 = np.vstack((y_test,y_test2))
        y_p = model.predict_proba(x_tr1)
        model_pred = y_p[:, 1]
        fpr_te1, tpr_te1, threholds = metrics.roc_curve(y_test1, model_pred)
        model_auc = metrics.auc(fpr_te1, tpr_te1)

        xtt = y_test1.reshape((-1,))
        xtp = model_pred.reshape((-1,))
        pp = []
        pn = []
        for i in range(len(xtt)):
            if xtt[i] > 0.5:
                pp.append(xtp[i])
            else:
                pn.append(xtp[i])

        tpr_te1, fpr_te1 = self.draw_roc(pp, pn)

        # print(p_len,n_len,type(pp))
        print(xtt.shape, xtp.shape)
        #y_dt = pd.DataFrame({'真实-测试': xtt, '预测测试': xtp})
        #y_dt.to_csv('y-dt用来画ROC.csv', index=False)

        y_dt = pd.DataFrame()
        y_dt['fp_rate'] = fpr_te1
        y_dt['tp_rate'] = tpr_te1
        y11 = pd.DataFrame(y_dt)
        y11.to_csv('./results/'+str(model_name)+'ROC_value.csv')

        #计算训练集ROC
        x_tra = np.array(x_train)

        yt_p = model.predict_proba(x_tra)
        yt_pred = yt_p[:, 1]
        yt_plable = model.predict(x_tra)
        yt_acc = metrics.accuracy_score(y_train, yt_plable)
        # 计算ROC,AUC
        fpr1, tpr1, threholds2 = metrics.roc_curve(y_train, yt_pred,drop_intermediate=False)
        #yt_auc = metrics.auc(fpr1, tpr1)

        #制作绘图数据
        x_tra2 = np.array(x_train2)
        x_tra1 = np.vstack((x_tra, x_tra2))
        y_train1 = np.vstack((y_train, y_train2))
        y_p = model.predict_proba(x_tra1)
        model_pred = y_p[:, 1]
        fpr_tr1, tpr_tr1, threholds = metrics.roc_curve(y_train1, model_pred)
        yt_auc = metrics.auc(fpr_tr1, tpr_tr1)



        y_dt = pd.DataFrame()
        y_dt['fp_rate'] = fpr_tr1
        y_dt['tp_rate'] = tpr_tr1
        y11 = pd.DataFrame(y_dt)
        y11.to_csv('./results/'+str(model_name) + '训练集ROC_value.csv')


        #画ROC曲线
        plt.figure(figsize=(7.8, 6), dpi=400)
        plt.plot(fpr_tr1, tpr_tr1, color='red', lw=2,label='训练集AUC = %0.3f' % yt_auc)
        plt.plot(fpr_te1, tpr_te1, color='darkorange', lw=2,label='测试集AUC = %0.3f' % model_auc)

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.legend(loc="lower right",prop={'size': 20})
        plt.xlabel('假正率', fontsize=20)
        plt.ylabel('真正率', fontsize=20)
        plt.xticks( size=18)
        plt.yticks(size=18)
        #plt.legend()
        TK = plt.gca()
        TK.spines['bottom'].set_linewidth(1.6)  # 图框下边
        TK.spines['left'].set_linewidth(1.6)  # 图框左边
        TK.spines['top'].set_linewidth(1.6)  # 图框上边
        TK.spines['right'].set_linewidth(1.6)
        plt.savefig('./results/ROC.jpg', dpi=400)

        #plt.show()

        return model_acc,model_auc,yt_auc



    def read_para(self,file_path, model_name, para_type):
        configuration = configparser.ConfigParser()
        configuration.read(file_path)
        model_para1 = configuration.get(model_name, para_type)
        para_list = model_para1.split(",")
        para_list = [para_list[i] for i in range(len(para_list))]
        jud = []
        #判断输入的数字还是字符
        for i in range(10):
            jud.append(para_list[0][0] == str(i))
        if True in jud:
            if '.' in para_list[0]:
                para_list = [float(para_list[i]) for i in range(len(para_list))]
            else:
                para_list = [int(para_list[i]) for i in range(len(para_list))]
        else:
            para_list = [para_list[i] for i in range(len(para_list))]

        if type(para_list[0]) != str:  #非字符处理方法
            if len(para_list) == 1:
                name_para = para_list
            elif len(para_list) > 1:
                para_inter = para_list[0]
                name_para = []
                while para_inter <= para_list[1]:
                    name_para.append(para_inter)
                    para_inter = para_inter + para_list[2]
        else:
            name_para = [para_list[i] for i in range(len(para_list))]
        return name_para

    def zone(self,model1,zone_lure):
        path = r'.\data\altitude.tif'
        model = joblib.load(model1)
        dem = self.read_tif_file(path)[2]
        dat1 = pd.read_csv(r'.\data\xtest_dataset.csv').fillna(0)
        #dat2 = pd.read_csv(r'.\data\xtrain_dataset.csv', header=None).fillna(0)
        sd1 = self.reduce_mem_usage(dat1)
        #sd2 = self.reduce_mem_usage(dat2)
        #data_X = pd.read_csv(r'.\data\xtest_dataset.csv', header=None)

        data_x = np.array(sd1)
        #sd1 = self.reduce_mem_usage(data_x[:2000000,:])
        #sd11 = self.reduce_mem_usage(data_x[2000000:,:])
        ss_x = StandardScaler()
        x1 = ss_x.fit_transform(self.d2)
        data_x = ss_x.transform(data_x)
        if len(data_x)>5000000:
            self.d1 = np.array(data_x[0:2000000:,:])
            sd111 = np.array(data_x[2000000:3500000:,:])
            sd112 = np.array(data_x[3500000:4800000,:])
            sd113 = np.array(data_x[4800000:,:])
        #self.d2 = np.array(sd2)
        #print(self.d1.shape)
        #if "XGBOOST1" in model1:
            #x_xgboost = np.array(pd.read_csv(r'./data/xtest_tree.csv',header=None))
            #x_xgboost = self.reduce_mem_usage(pd.read_csv(r'./data/xtest_dataset.csv',header=None))
            #y = model.predict_proba(x_xgboost)
        #else:
            #if self.d1.shape[0]>2000000:
            y1 = model.predict_proba(self.d1)
            y2 = model.predict_proba(sd111)
            y3 = model.predict_proba(sd112)
            y4 = model.predict_proba(sd113)
            y = np.vstack((y1,y2))
            y = np.vstack((y,y3))
            y = np.vstack((y, y4))
        else:
            y = model.predict_proba(data_x)
            #y = model.predict_proba(self.d1)

        #print(y[0])
        landslide_pro = y[:,1]
        landslide_pro = np.array(landslide_pro).reshape((dem.shape[0],dem.shape[1]))
        very_low,low,mide,high = 0,0,0,0
        #print(landslide_pro[500,600])
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):
                if landslide_pro[i][j] <=zone_lure[0]:
                    very_low+=1
                    #landslide_pro1[i][j] =0
                elif zone_lure[0] < landslide_pro[i][j] <=zone_lure[1]:
                    low+=1
                    #landslide_pro1[i][j] = 0.333
                elif zone_lure[1] < landslide_pro[i][j] <=zone_lure[2]:
                    mide+=1
                    #landslide_pro1[i][j] = 0.666
                else:
                    high+=1
                    #landslide_pro1[i][j] = 1
        very_low_rate = very_low/(dem.shape[0]*dem.shape[1])*100
        low_rate = low / (dem.shape[0] * dem.shape[1])*100
        mide_rate = mide / (dem.shape[0] * dem.shape[1])*100
        high_rate = high / (dem.shape[0] * dem.shape[1])*100

        #绘制分区图
        tras_POF = (1-landslide_pro) * 255
        data3 = np.where(np.isnan(self.scope), np.nan, tras_POF)
        data2 = np.reshape(data3, (tras_POF.shape[0], tras_POF.shape[1], 1))

        # img = Image.fromarray(np.uint8(data2))
        plt.figure(figsize=(6, 6), dpi=600)
        plt.axis('off')
        plt.imshow(data2,cmap = 'Spectral')
        #plt.axis('off')
        plt.savefig('./results/'+'pof_zone.jpg', dpi=600)
        return landslide_pro,high_rate,mide_rate ,low_rate ,very_low_rate



    def draw_roc(self,landslide, non_landslide):
        pa = np.linspace(0, 1, 1000)

        fn = []
        for k in range(1000):
            a1 = []
            for i in range(len(landslide)):
                if landslide[i] <= pa[k]:
                    a1.append(i)
            fn.append(len(a1))

        tn = []
        for k in range(1000):
            a1 = []
            for i in range(len(non_landslide)):
                if non_landslide[i] <= pa[k]:
                    a1.append(i)
            tn.append(len(a1))

        tpr, fpr = [], []
        for i in range(1000):
            tpr.append((len(landslide) - fn[i]) / len(landslide))
            fpr.append((len(non_landslide) - tn[i]) / len(non_landslide))

        return tpr, fpr

#app = QApplication([])

stats = Stats()
stats.ui.show()
# app.exec_()
sys.exit(app.exec_())