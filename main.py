from lib import *
from model import *
from nerusingcrf import *

# Mapping warna untuk setiap tag
TAG_COLOR_MAP = {
    'B-PER': 'red', 'I-PER': 'red',
    'B-ADJ': 'green', 'I-ADJ': 'green',
    'B-ANM': 'yellow', 'I-ANM': 'yellow',
    'B-GODS': 'blue', 'I-GODS': 'blue',
    'B-OBJ': 'purple', 'I-OBJ': 'purple',
    'O': 'black'
}

def colorize_text(word, tag):
    color = TAG_COLOR_MAP.get(tag, 'black')  # Default warna hitam
    return f"<span style='color:{color}'>{word}</span>"

def main():
    st.title("Final Projek NLP:blue[NER dengan Bahasa]:green[Bali]:stuck_out_tongue_closed_eyes::smile::scream::blush::angry::rage:")
    st.header('Conditional Random Field (CRF)')
    
    teks = st.text_input('Input text')
    
    if st.button('Prediksi Kalimat'):
        if teks:
            prediction = predict_text(teks)
            
            # Tampilkan hasil prediksi dengan warna
            colored_text = ' '.join([colorize_text(word, tag) for word, tag in prediction])
            st.markdown(f"<div style='font-size:18px'>{colored_text}</div>", unsafe_allow_html=True)
            
        else:
            st.write("Masukkan teks terlebih dahulu.")

if __name__ == '__main__':
    main()
