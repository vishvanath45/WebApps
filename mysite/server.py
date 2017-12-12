import os
import sys
import cv2
from skimage import io
import dlib
import numpy as np
from PIL import Image
from scipy import ndimage
from six.moves import cPickle as pickle
import openface
from resizeimage import resizeimage
import random
np.random.seed(133)
from flask import Flask, render_template, request

app = Flask(__name__)
#to do
UPLOAD_FOLDER = '/home/vnds_20150389/mysite2/images'

app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
@app.route('/')

def mainIndex():
    print ("starting server")
    return render_template('main_front.html')


@app.route('/upload',methods=['POST'])

def upload_file():
    file = request.files['image']
    
    file_name = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    file.save(file_name)
    
    modi = False
    ak = False
    person =False
    predictor_model = "./shape_predictor_68_face_landmarks.dat"

    face_detector = dlib.get_frontal_face_detector()

    face_aligner = openface.AlignDlib(predictor_model)

    image = io.imread(file_name)
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        os.remove('./static/initial.jpg') 
    except:
        pp = 'a'

    draw_image = cv2.imread(file_name)

    detected_faces = face_detector(image, 1)

    dataset = np.ndarray(shape = (1, 200*200*3),dtype = np.float32)

    for i, face_rect in enumerate(detected_faces):

       # alignedFace = face_aligner.align(534, image, face_rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOS)
       d = face_rect
        
       color1 = random.randint(0,255)
       color2 = random.randint(0,255)
       color3 = random.randint(0,255)

       cv2.rectangle(draw_image,(d.left(),d.top()),(d.right(),d.bottom()),(color1,color2,color3),1)

       image = io.imread(file_name)

       crop_img = face_aligner.align(534, image, face_rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

       mm = "initial.jpg"

       crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

       cv2.imwrite("./images/"+mm,crop_img)

       final_add = "./images/initial.jpg"

       with open('./images/initial.jpg','r+b') as f:
           with Image.open(f) as image:
               cover = resizeimage.resize_cover(image, [200,200],validate =False)
               cover.save(final_add,image.format)
        
       img_data = ((ndimage.imread('./images/initial.jpg').astype(float) - 255.0/2)/255.0)
       img_wd = 200
       img_ht = 200

       img_data = img_data.reshape(img_wd*img_ht*3)
       dataset[0, :] = img_data


       pickle_file_name = './Logistic_regression_fit.pickle'

       with open(pickle_file_name, 'rb') as f:
           model_from_pickle = pickle.load(f)
           predicc = model_from_pickle.predict(dataset)
           if (int(predicc) == int(1)):
               modi = True
               person =True
           elif (int(predicc) == int(2)):
               ak =True
               person = True
           elif (int(predicc) == int(3)):
               person = True
    

    if(os.path.exists('/static/2.jpg')):
        name_lol =1
        check_file_exists = True
    else:
        name_lol = 2
        check_file_exists =False

    name_lol =random.randint(1,400)

            
#    os.remove(file_name)
    display_file_name = './static/'+str(name_lol)+'.jpg'
    upload_file_name = 'static/'+str(name_lol)+'.jpg'
    cv2.imwrite(display_file_name ,draw_image)
    init  = True
    file_name = str(file_name).replace('/','',1)
    return render_template('main_page.html', **locals())


# starting server
if __name__ == '__main__':
    app.run(host = '0.0.0.0' , port = 80 , debug=True)
