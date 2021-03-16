
import streamlit as st
import pandas as pd
import numpy as np
#import tensorflow as tf
from tensorflow.keras import models
from skimage.transform import resize
from streamlit_drawable_canvas import st_canvas

# configurate layout
st.set_page_config(layout="wide",initial_sidebar_state="expanded")

# load tensorflow model
#@st.cache(allow_output_mutation=True) such that model stays in cache, and is not loaded again every time the user interacts with the UI
@st.cache(allow_output_mutation=True)
def load_model():
    model = models.load_model("tensorflow_CNN_number_guesser_model")
    model.make_predict_function()
    model.summary()  # included to make it visible when model is reloaded
    return model
    
if __name__ == "__main__":
    # title and description
    st.title("CNN Number Guesser")
    st.markdown("""
    This python application uses a simple convolutional neural network model (previously created and trained with tensorflow) to guess what number is drawn on the canvas below. 
    The model achieves more than 99 percent accuracy on the test data of the mnist dataset.
    """)

    expander_bar = st.beta_expander("About")
    expander_bar.markdown("""
    * **Python libraries: streamlit, streamlit_drawable_canvas, pandas, numpy, tensorflow, skimage**
    * **Source of data: MNIST dataset from tensorflow.keras.datasets, which contains 70,000 images (28x28 pixels) of the numbers from 0 to 9 handwritten by US highschool students.**
    """)
    st.write("")
    st.write("Try to draw a number here:")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=50,
        stroke_color="000000",
        background_color="#FFFFFF",
        background_image=None,
        update_streamlit=True,
        height=500,
        width=500,
        drawing_mode="freedraw",
        initial_drawing= None,
        key="full_app",
    )

    if st.button("Click here to let the CNN model make a prediction"):

        # convert the drawn canvas to image format and get a prediction from our pretrained model
        if canvas_result.image_data is not None:
            
            # load model
            tf_model = load_model()
            
            # convert canvas to 28x28 pixel image with greyscale values (from 0 to 1) to match training data, 0 means black/1 means white 
            input_img = (255 - canvas_result.image_data[:,:,:1])/255  
            input_img = resize(input_img, (28, 28))
            
            # calculate probability distribution for each number from 0 to 9 using our neural network
            prediction = tf_model.predict(np.array([input_img]))

            # get number with highest individual probability
            predicted_number = np.argmax(prediction[0])

            # return findings and probability distribution to user
            st.write(f"**Model guesses: {predicted_number}**")
            st.write(f"Among the numbers from 0 to 9 our model determines that our image has the highest resemblance to the samples of the training data with label {predicted_number}.")
            st.write("""
                Below is a table of the distribution with how likely our model evaluates the resemblance to each number from 0 to 9. 
                The sum of all values is 1. If a particular number receives a evaluation close to 1, the model determines that the image has the highest resemblance to training data with the same label (in comparison to all other numbers).
                Note that some numbers are drawn similarly (like 1/7 or 5/6) and some numbers are an unfinished version of another (e.g. 1 is a 4 that is still missing some lines). 
                In addition, drawing random lines, or drawing at the canvas border will result in incorrect guesses, as training data does not contain images where numbers cross the border or images with random noise/lines.
                The model guess should be interpreted as a mathematical evaluation on how close the drawn image is to the training data with the same label (and the only labels are the numbers from 0 to 9), 
                rather than a definitive judgement that is based on the geometry of the lines.
            """)
            st.write(pd.DataFrame(prediction))
    