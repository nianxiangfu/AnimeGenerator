import sys
from PyQt5.QtWidgets import QApplication,QMainWindow
from mainpage_plus import  Ui_MainWindow

# import tensorflow as tf
# from wgan.wgan import WGAN
# from ugan.ugan import UGAN
import cv2

if __name__ == '__main__':
    # tf.set_random_seed(9487)

    # wganSession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # wgan = WGAN(wganSession)
    # wgan.build_model()

    # uganSession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # ugan = UGAN(uganSession)
    # ugan.build_model()

    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(mainWindow)
    # ui.setupGan(ugan,wgan)
    mainWindow.show()
    sys.exit(app.exec_())