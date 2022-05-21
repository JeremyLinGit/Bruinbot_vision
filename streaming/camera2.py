import cv2
from firebase_admin import credentials, firestore, initialize_app, get_app

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
        self.db = self.init_with_service_account('./key/bruinbotv2.json')
        self.video = cv2.VideoCapture(0)
        if self.video.isOpened():
            print('video is open')
            self.success = True
        else:
            print('video didnt open :(')
            fb_data = {
                    'timestamp' : firestore.SERVER_TIMESTAMP,
                    'Location': 'streaming/camera.py',
                    'Message': 'Unable to open video capture' }
            self.db.collection('Logs').document().set(fb_data, merge=True)
            self.success = False
    def init_with_service_account(self, credentials_path):
        """
        Initialize the Firestore DB client using a service account
        :param file_path: path to service account
        :return: firestore
        """
        cred = credentials.Certificate(credentials_path)
        try:
            get_app()
        except ValueError:
            initialize_app(cred)
        return firestore.client()
    def __del__(self):
        self.video.release()
    def get_frame(self):
        self.success, image = self.video.read(1024)
        if self.success:
            image = cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()
# need to kill everything with command:
# kill -9 $(ps -A | grep python | awk '{print $1}')