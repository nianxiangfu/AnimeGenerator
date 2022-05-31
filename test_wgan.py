import tensorflow as tf
from wgan.wgan import WGAN
import cv2

tf.set_random_seed(9487)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

wgan = WGAN(sess)
wgan.build_model()
samples = wgan.generate("blonde hair", "blue eyes")
print(samples)
# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
#     wgan = WGAN(sess)
#     wgan.build_model()
#     samples = wgan.generate("blonde hair", "blue eyes")
#     print(samples)

