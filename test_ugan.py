import tensorflow as tf
from ugan.ugan import UGAN
import cv2

tf.set_random_seed(9487)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    ugan = UGAN(sess)
    ugan.build_model()
    samples = cv2.imread("ugan_test.jpg")
    print(samples)
    img = ugan.generate(samples)[0]

    
    print(img)
    cv2.imshow("1", img)
    cv2.waitKey()
