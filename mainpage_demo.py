# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainpage.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 381, 351))
        font = QtGui.QFont()
        font.setUnderline(False)
        self.label.setFont(font)
        self.label.setStyleSheet("border-color: rgb(255, 255, 127);\n"
"border-width: 1px;\n"
"border-style: solid;")
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setMidLineWidth(1)
        self.label.setObjectName("label")


        self.label.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.label.customContextMenuRequested.connect(self.rightMenuShow)


        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 400, 291, 80))
        self.groupBox.setObjectName("groupBox")


        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(150, 30, 110, 19))
        self.radioButton_2.setObjectName("radioButton_2")


        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(20, 30, 110, 19))
        self.radioButton.setObjectName("radioButton")


        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(480, 160, 93, 28))
        self.pushButton.setObjectName("pushButton")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.radioButton_2.setText(_translate("MainWindow", "RadioButton"))
        self.radioButton.setText(_translate("MainWindow", "RadioButton"))
        self.pushButton.setText(_translate("MainWindow", "PushButton"))

    def rightMenuShow(self):
        self.label.popMenu = QtWidgets.QMenu()
        Store=QtWidgets.QAction(u'保存', self.label)
        self.label.popMenu.addAction(Store)
        #绑定事件
        Store.triggered.connect(self.test)
        self.showContextMenu(self.label,QtGui.QCursor.pos())

    def test(self):
        print("申奥成功了")
        self.genarationImgae()
    def showContextMenu(self, label, pos):
        #调整位置
        # 菜单显示前，将它移动到鼠标点击的位置
        label.popMenu.move(pos)
        label.popMenu.show()

    
    def genarationImgae(self,param1 = 1,param2 = 2):
        """
        图片生成
        """
        image_path = './homework.png'
        picture = QtGui.QPixmap(image_path)
        # .scaled(self.label.width(), self.label.height())
        print(picture)
        self.label.setPixmap(picture)