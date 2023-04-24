import os 
import base64
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms.fields import FileField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
from PIL import Image
from io import BytesIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'I_hate_secret_keys'

# Load the labels
with open("ImageNetLabels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load the model from the local folder
model_path = "mobilenet_v2"
# model = tf.keras.models.load_model(model_path)
model = tf.keras.Sequential([
    hub.KerasLayer(model_path)
])
print(f"Model type: {type(model)}")



class UploadForm(FlaskForm):
    image = FileField('Upload an image (JPG format)', validators=[
        FileRequired(),
        FileAllowed(['jpg', 'jpeg'], 'Images only!')
    ])
    submit = SubmitField('Predict')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        image = form.image.data
        img = Image.open(image.stream).convert('RGB')
        
        # Preprocess the image
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_batch)
        predicted_class = np.argmax(predictions[0])

        # Get the label at the predicted class index
        predicted_label = labels[int(predicted_class)]

        # Display the result
        flash(f'Predicted class: {predicted_label}')

        return redirect(url_for('index'))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
