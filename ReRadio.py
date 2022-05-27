from PyQt5.QtWidgets import * 	
from PyQt5.QtGui import *		
from PyQt5.QtCore import *

class MyRadio(QRadioButton):
    def __init__(self,groupBox,self_plus):
        super().__init__(groupBox)
        self.clicked.connect(lambda:self.radioClick(groupBox,self_plus))
    
    def radioClick(self,groupBox,self_plus):      
        if(groupBox.title() == "头发颜色"):
            self_plus.select_1 = self.text()
        else:
            self_plus.select_2 = self.text()

        # print(groupBox.title())
