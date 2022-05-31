import sys
from PyQt5.QtWidgets import QApplication,QMainWindow
from mainpage_plus import  Ui_MainWindow

import tensorflow as tf
from wgan.wgan import WGAN
from ugan.ugan import UGAN
import cv2
from PyQt5.QtCore import QThread

import warnings
warnings.filterwarnings("ignore")

class Thread(QThread):
    def __init__(self,type):
        super(Thread, self).__init__()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if type == 'wgan':
            self.gan = WGAN(sess)
        elif type == 'ugan':
            self.gan = UGAN(sess)
        self.gan.build_model()
        pass
    
    def run(self):
        if type == 'wgan':
            self.gan = WGAN(sess)
        elif type == 'ugan':
            self.gan = UGAN(sess)
        pass


if __name__ == '__main__':
    tf.set_random_seed(9487)

    g1 = tf.Graph()
    g2 = tf.Graph()
    wganSession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),graph=g1)
    uganSession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g2)

    with wganSession.as_default():
        with g1.as_default():
            wgan = WGAN(wganSession)
            wgan.build_model()

    with uganSession.as_default():
        with g2.as_default():
            ugan = UGAN(uganSession)
            ugan.build_model()

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)

    ui.setupGan(ugan,wgan)
    mainWindow.show()

    samples = cv2.imread("ugan_test.jpg")
    # with wganSession.as_default():
    #     with wganSession.graph.as_default():
    #         print(wgan.generate("blonde hair", "blue eyes"))
    # print(uganSession.graph)
    # with uganSession.as_default():
    #     with uganSession.graph.as_default():
    #         print(ugan.generate(samples))
    
    sys.exit(app.exec_())
