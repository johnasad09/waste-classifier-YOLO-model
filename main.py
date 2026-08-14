from PIL import Image
import streamlit as st
from ultralytics import YOLO

# =======LOAD MODEL====== #
model = YOLO("./best.pt")

# =======PREDICTION FUNCTION====== #
def predict(image):
    results = model(image)
    result = results[0]
    pred_id = result.probs.top1
    label = result.names[pred_id]
    confidence = result.probs.top1conf
    return label, confidence


# =======STREAMLIT APP===== #
st.set_page_config(page_title='Waste Classifier', page_icon='🤖', layout='wide')

st.header('Waste Classifier YOLO Model')

col1, col2 = st.columns(2)

# =========COLUMN-1============ #
with col1:
    uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'jpeg', 'png'])
   
# =========COLUMN-2============ #
with col2:
    st.markdown('### Result')
    if uploaded_file is None:
        st.info('Please upload an image')
        
    else:
        image = Image.open(uploaded_file)
        st.image(image, width=200)
        
        label, confidence = predict(image)
        st.success(f'Predicted as {label}')
        st.metric(label='Confidence', value=f'{float(confidence):.2%}')
    
        
