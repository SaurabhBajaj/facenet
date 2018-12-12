""" Face recognition """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.spatial import distance
import numpy as np
import os
import cv2
import click
import pickle
import dlib
from camera import capture_image_from_camera

DETECTOR = dlib.get_frontal_face_detector()
WINDOW = dlib.image_window()
PREDICTOR_PATH = str(os.path.expanduser(
    "/Users/sbajaj/src/facenet/src/faceid/shape_predictor_68_face_landmarks.dat"))
SHAPE_PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(
    "/Users/sbajaj/src/facenet/src/faceid/dlib_face_recognition_resnet_model_v1.dat")


class FaceDB:
    def __init__(self, storage_path="/tmp/facedb/"):
        self.e = {}

    def add_embedding(self, name, embedding):
        self.e[name] = embedding.tolist()

    def list_embeddings(self):
        for name, emb in self.e.items():
            print(f"Mame:{name}\nEmbedding:{emb}")

    def remove_embeddings(self, name):
        pass

    def _get_euclidean_distance(self, val1, val2):
        return distance.euclidean(val1, val2)

    def find_match(self, embedding):
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        distances = []
        # print("ssb self.e.items() ", self.e.items())
        for name, emb in self.e.items():
            distance = self._get_euclidean_distance(
                np.array(emb), embedding)
            # print(f"distance with {name}: ", distance)
            distances.append((distance, name))
        return distances

    def print_distance(self, distances):
        for dist, name in distances:
            print(f"distance with {name}: ", dist)

    def return_closest_match(self, embedding):
        distances = self.find_match(embedding)
        distances = sorted(distances)
        return distances[0]

    def add_photo(self, name, img):
        pass


class FaceDBDriver:
    def __init__(self):
        pass

    def new_facedb(self):
        return FaceDB()

    def load_embeddings_from_disk(self, path="/tmp/face_embeddings"):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save_embeddings_to_disk(self, face_embedding,
                                path="/tmp/face_embeddings"):
        with open(path, 'wb') as f:
            pickle.dump(face_embedding, f)


def get_frame_from_camera():
    return camera.capture_image_from_camera()


def get_frame_from_file_path(path):
    """ returns numpy image array from a string file path """
    path = os.path.expanduser(path)
    return dlib.load_rgb_image(path)


def read_jpg_from_folder(path="/Users/sbajaj/src/facenet/data/team/"):
    filelist = []
    path = os.path.expanduser(path)
    for root, dirs, files in os.walk(path):
        for name in files:
            if '.jpg' in name or '.jpeg' in name:
                filelist.append(os.path.join(root, name))

    return filelist


def get_face_detections(frame):
    DETECTOR = dlib.get_frontal_face_detector()
    dets = DETECTOR(frame, 1)
    return dets


def get_aligned_face_chips(frame, det):
    predictor_path = str(os.path.expanduser(
        "/Users/sbajaj/src/facenet/src/faceid/shape_predictor_5_face_landmarks.dat"))
    shape_predictor = dlib.shape_predictor(predictor_path)
    face_list = dlib.full_object_detections()
    face_list.append(shape_predictor(frame, det))
    aligned_images = dlib.get_face_chips(frame, face_list, size=320)
    return aligned_images, face_list


def show_image_and_wait_for_enter(frame, dets=None):
    WINDOW.clear_overlay()
    WINDOW.set_image(frame)
    if dets:
        WINDOW.add_overlay(dets)
    dlib.hit_enter_to_continue()


def get_face_descriptor(frame, shape):
    """
    provide the complete frame and the final detected and aligned shape
    Get the face descriptor vectors as output
    """
    face_descriptor = facerec.compute_face_descriptor(frame, shape)
    return face_descriptor


def get_face_embedding(frame, face_obj_detection):
    face_embedding_vector = get_face_descriptor(frame, face_obj_detection)
    face_embedding = np.array(face_embedding_vector)
    return face_embedding


def gen_embedding_for_file_path(frame):
    show_image_and_wait_for_enter(frame)
    dets = get_face_detections(frame)
    face_embeddings = []
    for det in dets:
        chips, faces = get_aligned_face_chips(frame, det)
        show_image_and_wait_for_enter(chips[0])
        face = faces[0]
        face_embedding = get_face_embedding(frame, face)
        face_embeddings.append(face_embedding)
    return face_embeddings


def test_gen_embeddings(num=3):
    images_path = read_jpg_from_folder()
    images_path = images_path[:num]
    e = {}
    for image_path in images_path:
        person_name = os.path.basename(image_path)
        person_name = person_name.split(".jpg")[0]
        frame = get_frame_from_file_path(image_path)
        dets = get_face_detections(frame)
        for det in dets:
            # show_image_and_wait_for_enter(frame, det)
            chips, faces = get_aligned_face_chips(frame, det)
            face = faces[0]
            face_embedding = get_face_embedding(frame, face)
            e[person_name] = face_embedding
    return e


@click.group()
def cli():
    pass


@cli.command()
def populate_db():
    driver = FaceDBDriver()
    db = driver.new_facedb()
    embeddings = test_gen_embeddings(100)
    for name, e in embeddings.items():
        db.add_embedding(name, e)
    driver.save_embeddings_to_disk(db)


def _load_db():
    driver = FaceDBDriver()
    try:
        db = driver.load_embeddings_from_disk()
    except:
        db = FaceDB()
    return db


@cli.command()
def load_db():
    db = _load_db()
    print(db)


@cli.command()
@click.option("--name")
def register_face(name):
    driver = FaceDBDriver()
    db = _load_db()
    frame = capture_image_from_camera()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    embeddings = gen_embedding_for_file_path(frame)
    db.add_embedding(name, embeddings[0])
    driver.save_embeddings_to_disk(db)


@cli.command()
@click.option("-f", "--filename", "filename", default="")
@click.option("-n", "--name", "name", default="")
@click.option("--camera/--no-camera", default=True)
def facecompare(filename, name, camera):
    if not filename and camera is False:
        raise Exception("Either --filename or --camera argument is needed")
    db = _load_db()
    if filename:
        frame = get_frame_from_file_path(filename)
    else:
        frame = capture_image_from_camera()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (900, 600))
    embeddings = gen_embedding_for_file_path(frame)
    if not embeddings:
        print("Looks like no faces were deteceted!")
        exit(1)
    if name:
        print(f"comparing {name} face to other faces")
    # generate embedding
    # db.find_match(embeddings[0])
    distance, matchname = db.return_closest_match(embeddings[0])
    if distance > 0.6:
        print("Could not detect anyone")
        exit()
    print(f"Looks like this is {matchname}")


@cli.command()
def test():
    populate_db()
    load_db()


# cli.add_command(register)
# cli.add_command(detect_face)
# cli.add_command(test)
# cli.add_command(populate_db)
# cli.add_command(load_db)
# cli.add_command(facecompare)


if __name__ == '__main__':
    cli()
