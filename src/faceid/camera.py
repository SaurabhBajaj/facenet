""" Camera capture module """
import cv2


def capture_image_from_camera():
    """
    This will get access to the camera and will show the current camera output.
    On hitting c(click), it will capture that frame and return it
    On hitting, q, it will exit without capturing the camera frame
    """
    cap = cv2.VideoCapture(0)
    final_frame = None
    print("Please press the 'c' key to capture an image")
    while True:

        # Capture frame-by-frame
        _, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            _, frame = cap.read()
            final_frame = frame
            break

    cap.release()
    # cv2.destroyAllWindows()
    return final_frame


def main():
    """ Entrypoint """
    capture_image_from_camera()


if __name__ == '__main__':
    main()
