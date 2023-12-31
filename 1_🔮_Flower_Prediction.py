import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from predict import predict_flower
from tf_predict import tf_predict_flower
from streamlit_lottie import st_lottie


def main():
    st.set_page_config(
        page_title="FlowerResNet",
        page_icon="🔮",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "This Streamlit application showcases the classification of flower images using a custom ResNet model implemented in PyTorch. The Flowers-299 dataset, consisting of 299 flower classes, is utilized for training and evaluation. The application covers various aspects including data preprocessing, model training with SGD optimizer, validation, and model checkpointing for reusability."
        }
    )

    uploaded_file = None

    col1, col2 = st.columns([1, 4])

    # Lottie Animation
    with col1:
        animation_url = "https://assets5.lottiefiles.com/packages/lf20_Yz2AzPLPXW.json"
        animation = st_lottie(animation_url, height=170,
                              width=170, key="lottie")

    # Title
    with col2:
        st.title('Dual Model Flower Image Prediction')

    # Create language selection in the sidebar
    st.sidebar.title('Choose Language')
    language = st.sidebar.selectbox(
        'Select Language:', ('English', 'Spanish', 'French'))

    # Create sidebar
    st.sidebar.info(
        "An AI-powered flower image prediction app using PyTorch and Streamlit. Upload an image or provide an image URL to get instant predictions on flower species. Experience the magic of AI in a simple and intuitive interface!"
    )

    col3, col4 = st.columns(2)

    # Image Source Selector
    with col3:
        image_source = st.radio('Select Image Source:', ('Upload', 'URL'))
        model = st.selectbox('Select Model:', ('Pytorch', 'Tensorflow'))

    # Image Uploader
    with col4:
        if image_source == 'Upload':
            uploaded_file = st.file_uploader(
                'Choose an image file', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded Image',
                 use_column_width=True, width=300)

        # Predict button
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                if model != 'Tensorflow':
                    flower_name = predict_flower(image)
                    st.success(f'Predicted Flower: {flower_name}')
                else:
                    try:
                        flower_name = tf_predict_flower(image)
                        st.success(f'Predicted Flower: {flower_name}')
                    except Exception as e:
                        st.warning('Sorry somthing went wrong..', icon="🤖")
            st.empty()  # Remove the Lottie animation from the UI once prediction is complete

    elif image_source == 'URL':
        image_url = st.text_input('Enter Image URL:')

        if st.button('Predict'):
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))

                st.image(image, caption='Image from URL',
                         use_column_width=True, width=300)

                with st.spinner('Predicting...'):
                    if model != 'Tensorflow':
                        flower_name = predict_flower(image)
                        st.success(f'Predicted Flower: {flower_name}')
                    else:
                        try:
                            flower_name = tf_predict_flower(image)
                            st.success(f'Predicted Flower: {flower_name}')
                        except Exception as e:
                            st.warning('Sorry somthing went wrong..', icon="🤖")
                st.empty()  # Remove the Lottie animation from the UI once prediction is complete

            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == '__main__':
    main()
