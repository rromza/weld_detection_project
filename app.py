import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from model_utils import load_model, get_transforms, predict_image

# Настройки страницы
st.set_page_config(
    page_title="Weld Classifier",
    page_icon="🔬",
    layout="centered"
)

# CSS для минимального стиля
st.markdown("""
<style>
    .good {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .bad {
        color: #c62828;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .no-weld {
        color: #f57c00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .confidence {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_cached():
    """Загрузка модели с кэшированием"""
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, config, class_names = load_model("model_inference.pth", device)
        return model, config, class_names
    except:
        st.error("❌ Ошибка загрузки модели. Убедитесь, что файл model_inference.pth существует.")
        return None, None, None

def get_class_emoji(class_name):
    """Возвращает эмодзи для класса"""
    emojis = {
        "good_weld": "✅",
        "bad_weld": "❌", 
        "no_weld": "⚠️"
    }
    return emojis.get(class_name, "🔍")

def get_class_color(class_name):
    """Возвращает цвет для класса"""
    colors = {
        "good_weld": "good",
        "bad_weld": "bad",
        "no_weld": "no-weld"
    }
    return colors.get(class_name, "")

def plot_probabilities(class_names, probs):
    """Визуализация вероятностей с правильными цветами для каждого класса"""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Создаем список цветов в соответствии с порядком классов
    colors = []
    for class_name in class_names:
        if class_name == "good_weld":
            colors.append('#4CAF50')  # зеленый
        elif class_name == "bad_weld":
            colors.append('#F44336')  # красный
        elif class_name == "no_weld":
            colors.append('#FFC107')  # желтый
        else:
            colors.append('#9E9E9E')  # серый для неизвестных классов
    
    bars = ax.barh(class_names, probs, color=colors)
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Вероятность')
    ax.set_title('Вероятности классов')
    
    # Подписи значений
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
               f'{prob:.1%}', va='center')
    
    return fig

def get_class_display_name(class_name):
    """Возвращает читаемое название класса"""
    display_names = {
        "good_weld": "Качественный шов",
        "bad_weld": "Некачественный шов",
        "no_weld": "Шов не обнаружен"
    }
    return display_names.get(class_name, class_name)

def main():
    """Основная функция приложения"""
    
    # Заголовок
    st.title("🔬 Классификатор сварных швов")
    st.markdown("Загрузите изображение для анализа качества сварки")
    
    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model, config, class_names = load_model_cached()
    
    if model is None:
        return
    
    # Показываем порядок классов для отладки (можно закомментировать)
    st.sidebar.markdown("### Порядок классов:")
    for i, name in enumerate(class_names):
        st.sidebar.write(f"{i}. {name}")
    
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
                # Подготовка трансформаций
                transform = get_transforms(config['INPUT_SIZE'])
                
                # Предсказание
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                pred_class, confidence, all_probs = predict_image(
                    model, image, transform, device, class_names
                )
            
            # Отображение результата
            st.markdown("---")
            st.subheader("Результат:")
            
            # Эмодзи и название класса
            emoji = get_class_emoji(pred_class)
            color_class = get_class_color(pred_class)
            display_name = get_class_display_name(pred_class)
            
            # Отображение результата
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{emoji}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='{color_class}'>{display_name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='confidence'>Уверенность: {confidence:.1%}</div>", unsafe_allow_html=True)
            
            # Визуализация вероятностей
            st.markdown("---")
            st.subheader("Вероятности классов:")
            fig = plot_probabilities(class_names, all_probs)
            st.pyplot(fig)
            
            # Простая таблица с деталями
            with st.expander("Подробности"):
                st.write("**Техническая информация:**")
                st.write(f"- Устройство: {'GPU' if torch.cuda.is_available() else 'CPU'}")
                st.write(f"- Размер модели: {sum(p.numel() for p in model.parameters()):,} параметров")
                st.write(f"- Размер изображения: {config['INPUT_SIZE']}x{config['INPUT_SIZE']}")
                
                st.write("\n**Все вероятности:**")
                for name, prob in zip(class_names, all_probs):
                    if name == "good_weld":
                        icon = "✅"
                    elif name == "bad_weld":
                        icon = "❌"
                    elif name == "no_weld":
                        icon = "⚠️"
                    else:
                        icon = "🔍"
                    st.write(f"- {icon} {name}: {prob:.3%}")

    # Информация о классах (только при первом запуске)
    else:
        st.markdown("---")
        st.info("""
        **Информация о классах:**
        
        - ✅ **good_weld** - Качественный сварной шов
        - ❌ **bad_weld** - Некачественный сварной шов с дефектами
        - ⚠️ **no_weld** - Сварной шов не обнаружен на изображении
        """)

if __name__ == "__main__":
    main()
