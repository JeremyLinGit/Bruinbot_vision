from firebase_admin import credentials, firestore, get_app, initialize_app
import json
import os
def init_with_service_account(credentials_path):
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
if __name__ == '__main__':
    os.system("curl  http://localhost:4040/api/tunnels > tunnels.json")
    with open('tunnels.json') as data_file:
        datajson = json.load(data_file)
    URL = (datajson['tunnels'][0]['public_url'])
    db = init_with_service_account("./key/bruinbotv2.json")
    data = {
        "timestamp" : firestore.SERVER_TIMESTAMP,
        "url": URL
    }
    db.collection('NGROK').document("URL").set(data, merge=False)