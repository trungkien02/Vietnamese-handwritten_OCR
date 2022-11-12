import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
char_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789\
			 ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ\
			 ẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ "

threshold = [32, 128]
max_label_len = np.load('Data/npy-data/max_label_len.npy')

def encode_to_labels(txt):
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

def sque(x):
    return K.squeeze(x, axis = 1)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def convolutional_block(layer_in, filters, kernel_size, pool_size, strides, block):
    L2 = 1.0e-6
    # defining name basis
    conv_name_base = 'Convolution' + str(block) 
    bn_name_base = 'Batch' + str(block) 
    pool_name_base = 'Pool' + str(block) 

    X = Conv2D(filters = filters, kernel_size = kernel_size, activation = 'relu', padding = 'same', 
    		   name = conv_name_base, kernel_regularizer = regularizers.l2(L2))(layer_in) 
    X = BatchNormalization(name = bn_name_base)(X)
    X = Dropout(0.2)(X)
    layer_out = MaxPool2D(pool_size = pool_size, strides = strides, name = pool_name_base)(X)

    return layer_out

inputs = Input(shape=(threshold[0], threshold[1],1))

x = convolutional_block(layer_in=inputs, filters=64, kernel_size=(3,3), 
                            pool_size=(2,2), strides=2, block=1)

x = convolutional_block(layer_in=x, filters=128, kernel_size=(3,3), 
                            pool_size=(2,2), strides=2, block=2)

x = convolutional_block(layer_in=x, filters=256, kernel_size=(3,3), 
                            pool_size=(2,2), strides=2, block=3)
    
x = convolutional_block(layer_in=x, filters=256, kernel_size=(3,3), 
                            pool_size=(2,1), strides = 1, block=4)

x = convolutional_block(layer_in=x, filters=512, kernel_size=(3,3), 
                            pool_size=(2,1), strides=1, block=5)

x = convolutional_block(layer_in=x, filters=512, kernel_size=(3,3), 
                            pool_size=(2,1), strides=1, block=6)
    
squeezed = sque(x)
    
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25),
                            name = 'LSTM_1')(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25),
                            name='LSTM_2')(blstm_1)
    
outputs = Dense(len(char_list)+1, activation='softmax', name='fully_connected')(blstm_2)

act_model = Model(inputs, outputs)


labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

loss_out = Lambda(ctc_lambda_func, 
                  output_shape=(1,), 
                  name='ctc')([outputs, labels, input_length, label_length])

optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model = Model(inputs=[inputs, labels, input_length, label_length], 
                outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, 
                optimizer=optimizer)

model.load_weights('model/best_model.h5')
def resize_(img): 
    w, h = img.shape
    if (32/w < 128/h):
        scale_percent = 32/w
    else:
        scale_percent = 128/h
    height = int(img.shape[0] * scale_percent)
    width = int(img.shape[1] * scale_percent)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return img
    
def reshape_expand_dim_(img):
    w, h = img.shape
    if w < 32:
            add_zeros = np.ones((32-w, h))*210
            img = np.concatenate((img, add_zeros))
    if h < 128:
        add_zeros = np.ones((32, 128-h))*210
        img = np.concatenate((img, add_zeros), axis=1)
    img = np.expand_dims(img , axis = 2)
    print(img.shape)

    return img

    return img
def predict(img):

    img = img[None, :, :, :]
    prediction = act_model.predict(img)
    
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                             greedy=True)[0][0])
    predicted_str = ''
    for x in (out):
        for p in x:  
            if int(p) != -1:
                predicted_str += char_list[int(p)]    
    return predicted_str

def sort_contours(cnts, reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return cnts

st.title("OCR")
st.write("This is a simple OCR app")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    blur = cv2.GaussianBlur(img, (9,9), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sort_contours(contours, reverse=False)
    pred_str = ''
    for cnt in contours:       
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        roi = img[y:y+h, x:x+w]
        roi = resize_(roi)
        roi = reshape_expand_dim_(roi)
        roi = roi/255
        pred_str += predict(roi) + ' '
    st.write("Predicted string: ", pred_str)
