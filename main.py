import cv2

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def sign(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def deviation(faces, frame):
    if len(faces) > 0:
        large_face = max(faces, key=lambda face: face[2] * face[3])
        faces_center = (large_face[0] + large_face[0] + large_face[2]) / 2, (
                    large_face[1] + large_face[1] + large_face[3]) / 2

        width = frame.shape[1] / 2
        height = frame.shape[0] / 2

        dx = faces_center[0] - width
        dy = height - faces_center[1]

        print(dx, dy)

        if abs(dx) >= 60 or abs(dy) >= 60:
            cv2.putText(frame, f'dx: {sign(dx)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f'dy: {sign(dy)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f'Face reached!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return sign(dx), sign(dy)

    else:
        cv2.putText(frame, f'Face searching...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def camera_vision():
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        deviation(faces, frame)
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    camera_vision()