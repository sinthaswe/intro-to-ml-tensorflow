import tensorflow as tf
import tensorflow_hub as tfhub
import numpy as np
import json
import PIL
import argparse

img_size = 224

def process_img(img):
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (img_size, img_size))
    img /= 255
    img = img.numpy()
    return img

def predict(img_path, model, top_k):
    image = PIL.Image.open(img_path)
    image = np.asarray(image)
    image = process_img(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    k_values, k_indices = tf.math.top_k(prediction, top_k)
    k_values = k_values.numpy()
    k_indices = k_indices.numpy()
    return k_values, k_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('h5')
    parser.add_argument('top_k',type=int,default=5)
    parser.add_argument('label_map',type=str,default='label_map.json')
    args = parser.parse_args()

    with open(args.label_map, 'r') as f:
        label_map = json.load(f)

    img_path = args.img_path
    model = tf.keras.models.load_model(args.h5 ,custom_objects={'KerasLayer':tfhub.KerasLayer} )
    top_k = args.top_k
    k_values, k_indices = predict(img_path, model, top_k)
    probs = k_values.squeeze()

    print('\nClasses , Probabilities\n')

    j = 0
    for i in k_indices[0]:
        print(label_map[str(i+1)], ',' ,probs[j])
        j += 1

    print('\n')
