"""
Use the MTCNN model to work here and return the bounding boxes.
Uses the facenet mtcnn code here.
"""
# from facenet.contributed.predict import load_and_align_data
import tensorflow as tf
import facenet.src.align.detect_face as detect_face
from scipy import misc
import numpy as np
from src import facenet

import os


def load_and_align_data(frame, image_size, margin=40, gpu_memory_fraction=0.0):

    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    # nrof_samples = len(image_paths)
    img_list = []
    # img = misc.imread(os.path.expanduser(image_paths[i]))
    img = frame
    img_size = np.asarray(frame.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(
        img, minsize, pnet, rnet, onet, threshold, factor)
    img_list = []
    normalized_img_list = []
    num_faces = len(bounding_boxes)
    print("num_faces = ", num_faces)
    for j in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[j, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(
            cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        # import pdb
        # pdb.set_trace()
        img_list.append(aligned)
        normalized_img_list.append(prewhitened)
    # images = np.stack(img_list)
    # aligned_images =
    return img_list, normalized_img_list


if __name__ == '__main__':
    main()

# def get_bounding_boxes():
#     """ This will take a frame and then return the bounding boxes of the images """
#     pass

#     """Detects faces in an image, and returns bounding boxes and points for them.
#     img: input image
#     minsize: minimum faces' size
#     pnet, rnet, onet: caffemodel
#     threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
#     factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
#     """
