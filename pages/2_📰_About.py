import streamlit as st
from streamlit.components.v1 import html

# Trick to preserve the state of your widgets across pages
for k, v in st.session_state.items():
    st.session_state[k] = v

st.set_page_config(
        page_title="FlowerResNet|About",
        page_icon="🔮",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "This Streamlit application showcases the classification of flower images using a custom ResNet model implemented in PyTorch. The Flowers-299 dataset, consisting of 299 flower classes, is utilized for training and evaluation. The application covers various aspects including data preprocessing, model training with SGD optimizer, validation, and model checkpointing for reusability."
        }
    )

# Create language selection in the sidebar
st.sidebar.title('Choose Language')
language = st.sidebar.selectbox('Select Language:', ('English', 'Spanish', 'French'))

st.sidebar.info(
"An AI-powered flower image prediction app using PyTorch and Streamlit. Upload an image or provide an image URL to get instant predictions on flower species. Experience the magic of AI in a simple and intuitive interface!"
)


# Works with streamlit==1.17.0
# TODO: Review class names for future versions
st.markdown("""
  <style>
      ul[class="css-j7qwjs e1fqkh3o7"]{
        position: relative;
        padding-top: 2rem;
        display: flex;
        justify-content: center;
        flex-direction: column;
        align-items: center;
      }
      .css-17lntkn {
        font-weight: bold;
        font-size: 18px;
        color: grey;
      }
      .css-pkbazv {
        font-weight: bold;
        font-size: 18px;
      }
  </style>""", unsafe_allow_html=True)

st.header("About")

st.markdown(
    """
    ### **Brief description of the web app**
    This app is designed to assist in flower image prediction using a custom ResNet model implemented in PyTorch. Upload an image or provide an image URL to get instant predictions on flower species.
    ### **Features**
    - Upload an image or provide an image URL for prediction.
    - Custom ResNet model implemented in PyTorch.
    - Fast and accurate predictions on flower species.
    ### **Version**
    :sparkles: **Current version: 1.0** :sparkles:
    - Initial release.
    ### **Sources**
    - The source code of this app is available [here](https://github.com/000jd/FlowerResNet.git).
    ### **Message from the developer**
    > Dear users,
    > 
    > I am excited to present this flower prediction app, which aims to provide accurate predictions on flower species. I have developed this app using a custom ResNet model implemented in PyTorch to ensure high-quality results. 
    > 
    > I hope you find this app useful and enjoy using it. Your feedback and suggestions are always welcome!
    > 
    > Thank you for your support!
    > 
    > Best regards,
    > 
    > Your Name
    """, 
    unsafe_allow_html=True
)

html(f"""
    <a class="github-button" href="https://github.com/000jd/FlowerResNet.git" data-show-count="true" aria-label="Follow @your-username on GitHub">Follow @your-username</a>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
    <a class="twitter-follow-button" href="https://twitter.com/your-twitter-handle">Follow @your-twitter-handle</a>
    <script async defer src="https://platform.twitter.com/widgets.js"></script>
    """)
