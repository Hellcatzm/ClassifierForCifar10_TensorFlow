# Author : hellcat
# Time   : 18-6-19

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
 
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
import os
import numpy as np
import pprint as pp
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dict_label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
              5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
images = tf.concat([tf.expand_dims(
    tf.image.per_image_standardization(
        tf.image.resize_images(
            tf.image.decode_jpeg(
                open(os.path.join('./images', img), 'rb').read()), [24, 24], method=3)), axis=0)
    for img in list(os.walk('./images'))[0][2]], axis=0)

sess = tf.InteractiveSession(config=config)

ckpt = tf.train.get_checkpoint_state("./logs/model")
if ckpt is not None:
    print("[*] Success to read {}".format(ckpt.model_checkpoint_path))
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("[*] Failed to find a checkpoint")
# pp.pprint(tf.get_default_graph().get_operations())
image2show = sess.run(images)
g = tf.get_default_graph()
res = sess.run(g.get_tensor_by_name("logits:0"),
               feed_dict={g.get_tensor_by_name("Placeholder:0"): image2show})
print([dict_label[i] for i in np.argmax(res, axis=1)])
for i in range(1, 6):
    plt.subplot(1, 5, i)
    plt.imshow(image2show[i-1].astype(np.uint8))
    plt.xticks([])
    plt.yticks([])
plt.show()

# tf.train.start_queue_runners()
# images_show,  labels_show = sess.run([g.get_tensor_by_name("batch:0"),
#                                       g.get_tensor_by_name("Reshape_4:0")])
#
# for i in range(1, 11):
#     plt.subplot(2, 5, i)
#     plt.imshow(images_show[i-1].astype(np.uint8))
#     plt.title('{}'.format(labels_show[i-1]))
# plt.show()

