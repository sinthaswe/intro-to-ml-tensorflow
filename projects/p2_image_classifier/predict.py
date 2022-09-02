import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
from PIL import Image

image_size = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    im_arr = np.asarray(im)
    processed_im = process_image(im_arr)
    processed_im_batch = np.expand_dims(processed_im, axis=0)
    prediction = model.predict(processed_im_batch)
    probs, classes = tf.math.top_k(prediction,top_k)
    probs = probs.numpy().squeeze()
    classes_label = classes.numpy().squeeze()
    classes=[class_names[str(value+1)] for value in classes_label]

    return probs, classes


if __name__ == '__main__':
    print('predict.py, running')

    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('pretrained_model')
    parser.add_argument('--top_k',type=int,default=5)
    parser.add_argument('--category_names',default='label_map.json')

    args = parser.parse_args()
    print(args)
    print('arg1:', args.image_path)
    print('arg2:', args.pretrained_model)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    image_path = args.image_path
    model = tf.keras.models.load_model(args.pretrained_model ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k

    probs, classes = predict(image_path, model, top_k)

    print('Predicted Flower Name: \n',classes)
    print('Probabilities: \n ', probs)
