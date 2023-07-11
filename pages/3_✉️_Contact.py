import streamlit as st
import os

st.set_page_config(
    page_title="FlowerResNet|Contact",
    page_icon="🔮",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "This Streamlit application showcases the classification of flower images using a custom ResNet model implemented in PyTorch. The Flowers-299 dataset, consisting of 299 flower classes, is utilized for training and evaluation. The application covers various aspects including data preprocessing, model training with SGD optimizer, validation, and model checkpointing for reusability."
    }
)

# Trick to preserve the state of your widgets across pages
for k, v in st.session_state.items():
    st.session_state[k] = v

# Create language selection in the sidebar
st.sidebar.title('Choose Language')
language = st.sidebar.selectbox('Select Language:', ('English', 'Spanish', 'French'))

st.sidebar.info(
    "An AI-powered flower image prediction app using PyTorch and Streamlit. Upload an image or provide an image URL to get instant predictions on flower species. Experience the magic of AI in a simple and intuitive interface!"
)

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

st.header("Contact")

st.markdown(
    """
    Thank you for using our FlowerResNet application. If you have any questions, feedback, or inquiries, please feel free to contact us using the form below. We will get back to you as soon as possible.

    """
)

contact_form = """
<form action="https://formsubmit.co/{}" method="POST">
     <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
</form>
""".format(st.secrets["email"]["email_address"])

st.markdown(contact_form, unsafe_allow_html=True)

# Use Local CSS File
def local_css(file_name):
    path = os.path.dirname(__file__)
    file_name = path + "/" + file_name
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/email_style.css")
