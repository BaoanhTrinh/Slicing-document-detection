import imghdr
import os
from flask import Flask, render_template, request, redirect, url_for, abort, \
    send_from_directory
from werkzeug.utils import secure_filename
import utils

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4098 *  4098
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = ''
NUM_FEATURES = 75
def validate_image(stream):
    header = stream.read(512)  # 512 bytes should be enough for a header check
    stream.seek(0)  # reset stream pointer
    format = imghdr.what(None, header)
    if not format:
        return None
    return '.' + (format if format != 'jpeg' else 'jpg')

def predict(filename): 
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn import metrics
    import numpy
    model = utils.build_model()
    import time
    start = time.time()
    features = utils.get_features(filename)
    model.load_weights("models/model")
    X = features.reshape((1,NUM_FEATURES,1))
    y_pred = model.predict(X)
    end = time.time()
    return numpy.round(y_pred[0]), end-start

@app.route('/<filename>')
def index(filename):
    lst = filename.split(':')
    if (len(lst) > 1):
        path = app.config['UPLOAD_PATH'] + '/' + str(lst[0])
        if (int(lst[-2]) == 1):
            return render_template('index.html', file= path,true_label= path[1:7], predicted_label= 'normal', reference_time = lst[-1])
        else:
            return render_template('index.html', file= path,true_label= path[1:8], predicted_label= 'slicing', reference_time = lst[-1])
    else:
        return render_template('index.html',file = "non",true_label="non", predicted_label="non", reference_time = '0.0')

@app.route('/<filename>', methods=['POST'])
def upload_files(filename):
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS'] or \
                file_ext != validate_image(uploaded_file.stream):
            abort(400)
        path = os.path.join('static',filename)
        uploaded_file.save(path)
        predict_label,ref_time = predict(path)
        filename = filename+':' + str(int(predict_label)) + ':' +str(ref_time)
    return redirect(url_for('index',filename = filename))

@app.route('/uploads/<filename>')
def upload(filename):
    path = filename.split('/')
    return send_from_directory(app.config['UPLOAD_PATH'], path[-1]) 

@app.route('/references')
def references():
   return render_template('references.html')
   
if __name__ == '__main__':
   app.run(debug = True)