import cv2
from scipy import ndimage
from detector import gen_embedding_for_file_path, _load_db


def add_text():
    pass


def get_face_box():
    pass


def get_person_in_the_box():
    pass


def generate_final_output():
    pass


def demo_inference():
    # cap = cv2.VideoCapture('/Users/saurabh/Documents/saurabh-door.MOV')
    cap = cv2.VideoCapture('/Users/saurabh/Documents/emma-door.MOV')
    i = -1
    db = _load_db()
    face_detected = 0
    face_not_detected = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = ndimage.rotate(frame, -90)
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        i += 1
        if i % 3 != 0:
            continue
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        embeddings = gen_embedding_for_file_path(frame, show=False)
        if embeddings:
            distance, matchname = db.return_closest_match(embeddings[0])
            print(f"Match name = {matchname}")
            face_detected += 1
        else:
            print("no face detected")
            face_not_detected += 1
    # run face detection
    # run comparison with faces
    # generate final window with the face and identity
    print(
        f"Number of face detected: {face_detected}, not detected = {face_not_detected}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo_inference()
