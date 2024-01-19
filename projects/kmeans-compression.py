import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
import warnings

plt.style.use('ggplot'); sns.set()

warnings.filterwarnings('ignore')


def normalize(image):
    return image / 255

def flatten(image):
    return image.reshape(-1, image.shape[-1])

def compress_image(image, num_colors=16):
    image = normalize(image)
    flat_image = flatten(image)
    kmeans = MiniBatchKMeans(num_colors)
    kmeans.fit(flat_image)
    new_colors = kmeans.cluster_centers_[kmeans.predict(flat_image)]
    image_recolored = new_colors.reshape(image.shape)

    return image_recolored

def main():
    st.title('Image Compression using K-Means Clustering')
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = io.imread(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
        num_colors = st.slider('Select the number of colors:', 2, 256, 16)

        compressed_image = compress_image(image, num_colors)

        st.image(compressed_image, caption=f'Compressed Image with {num_colors} colors', use_column_width=True)


if __name__ == '__main__':
    main()