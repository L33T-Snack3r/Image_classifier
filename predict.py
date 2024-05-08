import argparse
import tensorflow as tf
from PIL import Image
import numpy as np
import json

def load_model(filepath):
    """
    Loads the model
    """
    return tf.keras.models.load_model(filepath)

def load_category_names(filepath):
    """
    Function to load category names from a JSON file.    
    """
    with open(filepath, 'r') as f:
        return json.load(f)    
    
def process_image(image):
    """
    Shapes the input image appropriately
    Input:
        - image - numpy array
    Output:
        image.numpy() - numpy array 
    
    """
    image_size = 224
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k=5):
    im = Image.open(image_path)
    image_array = np.asarray(im)
    
    #preprocess
    image_array = process_image(image_array)
    
    #predict
    all_probabilities = model.predict(np.expand_dims(image_array,axis=0))
    classes = all_probabilities.argsort()[0][-top_k:][::-1]
    
    #final output
    probabilities = all_probabilities[:,classes].tolist()[0]
    classes = list(map(str, classes.tolist()))
    
    return probabilities, classes

def load_category_names(filepath):
    # Function to load category names from a JSON file.
    import json
    with open(filepath, 'r') as f:
        return json.load(f)

def make_prediction(image_path, model_name, top_k=None, category_names=None):    
    #load model
    model = load_model(args.model_name)
    
    #Make load image, make prediction
    if top_k is not None:
        probabilities, classes = predict(image_path, model, top_k)
    else:
        probabilities, classes = predict(image_path, model)
    
    #categories
    if category_names is not None:
        categories = load_category_names(category_names)
        classes = [str(int(cls) + 1) for cls in classes]
        classes = [categories[cls] for cls in classes]
    
    for i in range(len(classes)):
        print("{} : {}".format(classes[i],probabilities[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict from a given model.")
    parser.add_argument('image_path', type=str, help='Path to the file')
    parser.add_argument('model_name', type=str, help='Input file name')
    parser.add_argument('--top_k', type=int, help='Return top K predictions', default=None)
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories', default=None)
    
    args = parser.parse_args()
    
    make_prediction(args.image_path, args.model_name, args.top_k, args.category_names)