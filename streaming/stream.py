
from flask import Flask, render_template, Response
from camera2 import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
    #return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
'''
# unnecessary codes
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5014, debug=False)
