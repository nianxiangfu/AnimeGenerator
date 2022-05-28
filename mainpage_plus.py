# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from Relabel import MyLabel
from ReRadio import MyRadio



class Ui_MainWindow(object):
    
    select_1 = 0
    select_2 = 0
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1355, 912)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(0, 0, 200, 862))
        self.groupBox_3.setStyleSheet("background-color: rgb(255, 255, 0);")
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.to_another = QtWidgets.QPushButton(self.groupBox_3)
        self.to_another.setGeometry(QtCore.QRect(0, 270, 201, 51))
        self.to_another.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.to_another.setObjectName("to_another")
        self.to_another_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.to_another_2.setGeometry(QtCore.QRect(0, 400, 201, 51))
        self.to_another_2.setStyleSheet("background-color: rgb(255, 170, 0);")
        self.to_another_2.setObjectName("to_another_2")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(200, 0, 1141, 862))
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.picture_6 = MyLabel(self.groupBox_4)
        self.picture_6.setGeometry(QtCore.QRect(890, 270, 151, 151))
        self.picture_6.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_6.setObjectName("picture_6")

        self.pushButton = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton.setGeometry(QtCore.QRect(130, 160, 93, 28))
        self.pushButton.setObjectName("pushButton")

        self.pushButton.clicked.connect(self.generation)

        self.picture_7 = MyLabel(self.groupBox_4)
        self.picture_7.setGeometry(QtCore.QRect(380, 530, 151, 151))
        self.picture_7.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_7.setObjectName("picture_7")

        self.picture_5 = MyLabel(self.groupBox_4)
        self.picture_5.setGeometry(QtCore.QRect(630, 270, 151, 151))
        self.picture_5.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_5.setObjectName("picture_5")

        self.picture_8 = MyLabel(self.groupBox_4)
        self.picture_8.setGeometry(QtCore.QRect(630, 530, 151, 151))
        self.picture_8.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_8.setObjectName("picture_8")
        self.picture_3 = MyLabel(self.groupBox_4)
        self.picture_3.setGeometry(QtCore.QRect(890, 30, 151, 151))
        self.picture_3.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_3.setObjectName("picture_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_4)
        self.groupBox_2.setGeometry(QtCore.QRect(190, 200, 151, 581))
        self.groupBox_2.setObjectName("groupBox_2")
        self.rButton_eyepurple = MyRadio(self.groupBox_2,self)
        self.rButton_eyepurple.setGeometry(QtCore.QRect(30, 310, 115, 19))
        self.rButton_eyepurple.setObjectName("rButton_eyepurple")
        self.rButton_eyeblack = MyRadio(self.groupBox_2,self)
        self.rButton_eyeblack.setGeometry(QtCore.QRect(30, 70, 115, 19))
        self.rButton_eyeblack.setObjectName("rButton_eyeblack")
        self.rButton_eyegreen = MyRadio(self.groupBox_2,self)
        self.rButton_eyegreen.setGeometry(QtCore.QRect(30, 190, 115, 19))
        self.rButton_eyegreen.setObjectName("rButton_eyegreen")
        self.rButton_eyebrown = MyRadio(self.groupBox_2,self)
        self.rButton_eyebrown.setGeometry(QtCore.QRect(30, 150, 115, 19))
        self.rButton_eyebrown.setObjectName("rButton_eyebrown")
        self.rButton_eyeyellow = MyRadio(self.groupBox_2,self)
        self.rButton_eyeyellow.setGeometry(QtCore.QRect(30, 390, 115, 19))
        self.rButton_eyeyellow.setObjectName("rButton_eyeyellow")
        self.rButton_eyeaqua = MyRadio(self.groupBox_2,self)
        self.rButton_eyeaqua.setGeometry(QtCore.QRect(30, 30, 115, 19))
        self.rButton_eyeaqua.setObjectName("rButton_eyeaqua")
        self.rButton_eyepink = MyRadio(self.groupBox_2,self)
        self.rButton_eyepink.setGeometry(QtCore.QRect(30, 270, 115, 19))
        self.rButton_eyepink.setObjectName("rButton_eyepink")
        self.rButton_eyeorange = MyRadio(self.groupBox_2,self)
        self.rButton_eyeorange.setGeometry(QtCore.QRect(30, 230, 115, 19))
        self.rButton_eyeorange.setObjectName("rButton_eyeorange")
        self.rButton_eyeblue = MyRadio(self.groupBox_2,self)
        self.rButton_eyeblue.setGeometry(QtCore.QRect(30, 110, 115, 19))
        self.rButton_eyeblue.setObjectName("rButton_eyeblue")
        self.rButton_eyered = MyRadio(self.groupBox_2,self)
        self.rButton_eyered.setGeometry(QtCore.QRect(30, 350, 115, 19))
        self.rButton_eyered.setObjectName("rButton_eyered")
        self.picture_2 = MyLabel(self.groupBox_4)
        self.picture_2.setGeometry(QtCore.QRect(630, 30, 151, 151))
        self.picture_2.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_2.setObjectName("picture_2")
        self.picture_4 = MyLabel(self.groupBox_4)
        self.picture_4.setGeometry(QtCore.QRect(380, 280, 151, 151))
        self.picture_4.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_4.setObjectName("picture_4")
        self.picture_9 = MyLabel(self.groupBox_4)
        self.picture_9.setGeometry(QtCore.QRect(890, 530, 151, 151))
        self.picture_9.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_9.setObjectName("picture_9")
        self.picture_1 = MyLabel(self.groupBox_4)
        self.picture_1.setGeometry(QtCore.QRect(380, 30, 151, 151))
        self.picture_1.setFrameShape(QtWidgets.QFrame.Box)
        self.picture_1.setObjectName("picture_1")
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_4)
        self.groupBox.setGeometry(QtCore.QRect(30, 200, 151, 581))
        self.groupBox.setObjectName("groupBox")
        self.rButton_hairorange = MyRadio(self.groupBox,self)
        self.rButton_hairorange.setGeometry(QtCore.QRect(30, 310, 115, 19))
        self.rButton_hairorange.setObjectName("rButton_hairorange")
        self.rButton_hairblack = MyRadio(self.groupBox,self)
        self.rButton_hairblack.setGeometry(QtCore.QRect(30, 70, 115, 19))
        self.rButton_hairblack.setObjectName("rButton_hairblack")
        self.rButton_hairbrown = MyRadio(self.groupBox,self)
        self.rButton_hairbrown.setGeometry(QtCore.QRect(30, 190, 115, 19))
        self.rButton_hairbrown.setObjectName("rButton_hairbrown")
        self.rButton_hairblue = MyRadio(self.groupBox,self)
        self.rButton_hairblue.setGeometry(QtCore.QRect(30, 150, 115, 19))
        self.rButton_hairblue.setObjectName("rButton_hairblue")
        self.rButton_hairpurple = MyRadio(self.groupBox,self)
        self.rButton_hairpurple.setGeometry(QtCore.QRect(30, 390, 115, 19))
        self.rButton_hairpurple.setObjectName("rButton_hairpurple")
        self.rButton_hairaqua = MyRadio(self.groupBox,self)
        self.rButton_hairaqua.setGeometry(QtCore.QRect(30, 30, 115, 19))
        self.rButton_hairaqua.setObjectName("rButton_hairaqua")
        self.rButton_hairgreen = MyRadio(self.groupBox,self)
        self.rButton_hairgreen.setGeometry(QtCore.QRect(30, 270, 115, 19))
        self.rButton_hairgreen.setObjectName("rButton_hairgreen")
        self.rButton_hairgrey = MyRadio(self.groupBox,self)
        self.rButton_hairgrey.setGeometry(QtCore.QRect(30, 230, 115, 19))
        self.rButton_hairgrey.setObjectName("rButton_hairgrey")
        self.rButton_hairblonde = MyRadio(self.groupBox,self)
        self.rButton_hairblonde.setGeometry(QtCore.QRect(30, 110, 115, 19))
        self.rButton_hairblonde.setObjectName("rButton_hairblonde")
        self.rButton_hairpink = MyRadio(self.groupBox,self)
        self.rButton_hairpink.setGeometry(QtCore.QRect(30, 350, 115, 19))
        self.rButton_hairpink.setObjectName("rButton_hairpink")
        self.rButton_hairred = MyRadio(self.groupBox,self)
        self.rButton_hairred.setGeometry(QtCore.QRect(30, 430, 115, 19))
        self.rButton_hairred.setObjectName("rButton_hairred")
        self.rButton_hairwhite = MyRadio(self.groupBox,self)
        self.rButton_hairwhite.setGeometry(QtCore.QRect(30, 470, 115, 19))
        self.rButton_hairwhite.setObjectName("rButton_hairwhite")

        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(200, 0, 1141, 862))
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.label = QtWidgets.QLabel(self.groupBox_5)
        self.label.setGeometry(QtCore.QRect(120, 130, 241, 281))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setGeometry(QtCore.QRect(580, 130, 241, 281))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_2.setGeometry(QtCore.QRect(170, 490, 121, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_3.setGeometry(QtCore.QRect(410, 230, 121, 71))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_4.setGeometry(QtCore.QRect(650, 490, 121, 71))
        self.pushButton_4.setObjectName("pushButton_4")
        self.groupBox_4.raise_()
        self.groupBox_3.raise_()
        self.groupBox_5.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1355, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.to_another.clicked.connect(lambda: self.to_show(1))
        self.to_another_2.clicked.connect(lambda: self.to_show(2))
        self.groupBox_5.hide()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.to_another.setText(_translate("MainWindow", "头像生成->"))
        self.to_another_2.setText(_translate("MainWindow", "风格迁移->"))
        self.picture_6.setText(_translate("MainWindow", "图片六"))

        self.pushButton.setText(_translate("MainWindow", "生成"))
        self.picture_7.setText(_translate("MainWindow", "图片七"))
        self.picture_5.setText(_translate("MainWindow", "图片五"))

        self.picture_8.setText(_translate("MainWindow", "图片八"))
        self.picture_3.setText(_translate("MainWindow", "图片三"))
        self.groupBox_2.setTitle(_translate("MainWindow", "眼睛颜色"))
        self.rButton_eyepurple.setText(_translate("MainWindow", "紫色"))
        self.rButton_eyeblack.setText(_translate("MainWindow", "黑色"))
        self.rButton_eyegreen.setText(_translate("MainWindow", "绿色"))
        self.rButton_eyebrown.setText(_translate("MainWindow", "棕黄色"))
        self.rButton_eyeyellow.setText(_translate("MainWindow", "黄色"))
        self.rButton_eyeaqua.setText(_translate("MainWindow", "浅绿色"))
        self.rButton_eyepink.setText(_translate("MainWindow", "粉色"))
        self.rButton_eyeorange.setText(_translate("MainWindow", "橘黄色"))
        self.rButton_eyeblue.setText(_translate("MainWindow", "蓝色"))
        self.rButton_eyered.setText(_translate("MainWindow", "红色"))
        self.picture_2.setText(_translate("MainWindow", "图片二"))
        self.picture_4.setText(_translate("MainWindow", "图片四"))
        self.picture_9.setText(_translate("MainWindow", "图片九"))
        self.picture_1.setText(_translate("MainWindow", "图片一"))
        self.groupBox.setTitle(_translate("MainWindow", "头发颜色"))
        self.rButton_hairorange.setText(_translate("MainWindow", "橘黄色"))
        self.rButton_hairblack.setText(_translate("MainWindow", "黑色"))
        self.rButton_hairbrown.setText(_translate("MainWindow", "棕黄色"))
        self.rButton_hairblue.setText(_translate("MainWindow", "蓝色"))
        self.rButton_hairpurple.setText(_translate("MainWindow", "紫色"))
        self.rButton_hairaqua.setText(_translate("MainWindow", "浅绿色"))
        self.rButton_hairgreen.setText(_translate("MainWindow", "绿色"))
        self.rButton_hairgrey.setText(_translate("MainWindow", "灰色"))
        self.rButton_hairblonde.setText(_translate("MainWindow", "金黄色"))
        self.rButton_hairpink.setText(_translate("MainWindow", "粉色"))
        self.rButton_hairred.setText(_translate("MainWindow", "红色"))
        self.rButton_hairwhite.setText(_translate("MainWindow", "白色"))
        self.label.setText(_translate("MainWindow", "图片1"))
        self.label_2.setText(_translate("MainWindow", "图片2"))
        self.pushButton_2.setText(_translate("MainWindow", "上传"))
        self.pushButton_3.setText(_translate("MainWindow", "转换->"))
        self.pushButton_4.setText(_translate("MainWindow", "保存"))

    def generation(self):
        print("申奥成功了")   
        print("select_1"+self.select_1)
        print("select_2"+self.select_2)

    def to_show(self,type):
        if(type == 1):
            self.groupBox_4.show()
            self.groupBox_5.hide()
        else:
            self.groupBox_4.hide()
            self.groupBox_5.show()
