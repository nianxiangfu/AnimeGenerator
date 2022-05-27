from PyQt5 import QtGui,QtWidgets,QtCore
from PyQt5.QtWidgets import * 	
from PyQt5.QtGui import *		
from PyQt5.QtCore import *
from PIL import Image
import numpy as np
import cv2

class MyLabel(QLabel):
    def __init__(self,centralwidget):
        super().__init__(centralwidget)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.rightMenuShow)
        # image_path = './homework.png'
        # picture = QtGui.QPixmap(image_path).scaled(self.width(), self.height())
        # self.setPixmap(picture)

    def rightMenuShow(self, point):
            self.popMenu = QMenu()
            store=QAction(u'保存', self)
            self.popMenu.addAction(store)
            store.triggered.connect(self.click_event)
            self.showContextMenu(QtGui.QCursor.pos())

    def click_event(self):
        self.read_file()

    def read_file(self):
        im = Image.open("./homework.png")  # 打开图片
        im_array = np.array(im)  # 将图片转化为numpy数组

        print(im_array)


        img_pil = Image.fromarray(np.uint8(im_array))
        
        img_pix = img_pil.toqpixmap().scaled(self.width(), self.height())
        
        self.setPixmap(img_pix)
        cv2.imwrite("out_cv2.jpg", im_array)  # 保存图片
        
    def showContextMenu(self, pos):
        self.popMenu.move( pos)
        self.popMenu.show()

