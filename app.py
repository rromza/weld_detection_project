import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model_utils import load_model, get_transforms, predict_image

# Настройки страницы
st.set_page_config(
    page_title="Определение качества сварочного шва",
    layout="centered"
)

# Простой CSS
st.markdown("""
<style>
    .good { color: green; font-weight: bold; }
    .bad { color: red; font-weight: bold; }
    .no-weld { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cached():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, config, class_names = load_model("model_inference.pth", device)
    return model, config, class_names

def get_class_emoji(class_name):
    emojis = {
        "good_weld": "✅",
        "bad_weld": "❌", 
        "no_weld": "⚠️"
    }
    return emojis.get(class_name, "🔍")

def get_class_color(class_name):
    colors = {
        "good_weld": "good",
        "bad_weld": "bad",
        "no_weld": "no-weld"
    }
    return colors.get(class_name, "")

def plot_probabilities(class_names, probs):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['red', 'green', 'orange'][:len(class_names)]
    bars = ax.barh(class_names, probs, color=colors)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Вероятность')
    ax.set_title('Вероятности классов')
    
    # Подписи значений
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{prob:.1%}', va='center')
    
    return fig

def main():
    
    st.title("Классификатор сварных швов")
    st.markdown("Загрузите изображение для анализа качества сварки")
    
    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model, config, class_names = load_model_cached()
        
    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['jpg', 'jpeg', 'png'],
        help="Поддерживаются JPG, PNG"
    )
    
    if uploaded_file is not None:
        # Показ изображения
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Загруженное изображение", use_column_width=True)
        except Exception as e:
            st.error(f"Ошибка загрузки изображения: {e}")
            return
        
        # Кнопка анализа
        if st.button("Анализировать", type="primary"):
            with st.spinner("Анализ..."):
                transform = get_transforms()
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                pred_class, confidence, all_probs = predict_image(
                    model, image, transform, device, class_names
                )
            
            st.markdown("---")
            st.subheader("Результат:")
            
            emoji = get_class_emoji(pred_class)
            color_class = get_class_color(pred_class)
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"## {emoji}")
            with col2:
                st.markdown(f"### <span class='{color_class}'>{pred_class}</span>", unsafe_allow_html=True)
                st.write(f"Уверенность: **{confidence:.1%}**")
            
            st.markdown("---")
            st.subheader("Вероятности классов:")
            fig = plot_probabilities(class_names, all_probs)
            st.pyplot(fig)
            
    else:
        st.markdown("---")
        st.info("""
        **Информация о классах:**
        
        - ✅ **good_weld** - Качественный сварной шов
        - ❌ **bad_weld** - Некачественный сварной шов с дефектами
        - ⚠️ **no_weld** - Сварной шов не обнаружен на изображении
        """)


main()
