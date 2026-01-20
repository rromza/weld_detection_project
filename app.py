import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import load_model, get_transforms, predict_image, load_and_prepare_image
import time

# Настройки страницы
st.set_page_config(
    page_title="Weld Quality Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для улучшения отображения
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .prediction-good {
        padding: 1rem;
        background-color: #C8E6C9;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .prediction-bad {
        padding: 1rem;
        background-color: #FFCDD2;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 20px;
        background-color: #E0E0E0;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        text-align: center;
        color: white;
        line-height: 20px;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_model():
    """Инициализация модели с кэшированием"""
    # Путь к модели (можете изменить при необходимости)
    model_path = "model_inference.pth"
    
    # Проверка наличия файла
    if not os.path.exists(model_path):
        st.error(f"❌ Файл модели не найден: {model_path}")
        st.info("Пожалуйста, убедитесь, что файл model_inference.pth находится в той же директории")
        return None, None, None
    
    # Определяем устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Загружаем модель
        model, config, class_names = load_model(model_path, device)
        return model, config, class_names
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке модели: {str(e)}")
        return None, None, None

def main():
    """Основная функция приложения"""
    
    # Заголовок приложения
    st.markdown('<h1 class="main-header">🔬 Weld Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    Это приложение использует обученную нейронную сеть для классификации качества сварных швов.
    Загрузите изображение сварного шва, и модель определит, является ли он качественным или нет.
    """)
    
    # Инициализация модели
    with st.spinner("🔄 Загрузка модели..."):
        model, config, class_names = initialize_model()
    
    if model is None:
        return
    
    # Сайдбар с информацией
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/welder.png", width=100)
        st.markdown("### 📊 Информация о модели")
        st.info(f"**Модель:** {config['MODEL_NAME']}")
        st.info(f"**Классы:** {', '.join(class_names)}")
        st.info(f"**Устройство:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
        st.info(f"**Размер входного изображения:** {config['INPUT_SIZE']}x{config['INPUT_SIZE']}")
        
        st.markdown("---")
        st.markdown("### 📝 Руководство")
        st.markdown("""
        1. Загрузите изображение сварного шва
        2. Модель автоматически проанализирует его
        3. Посмотрите результат и уверенность модели
        4. Используйте примеры для тестирования
        """)
        
        st.markdown("---")
        st.markdown("### 🔍 Примеры классов")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Качественный шов")
            st.caption("Гладкий, равномерный, без дефектов")
        with col2:
            st.error("❌ Некачественный шов")
            st.caption("Поры, трещины, неровности")
    
    # Основная область
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📤 Загрузка изображения</h3>', unsafe_allow_html=True)
        
        # Варианты загрузки
        upload_option = st.radio(
            "Выберите способ загрузки:",
            ["Загрузить файл", "Использовать URL", "Примеры изображений"]
        )
        
        image = None
        
        if upload_option == "Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                help="Поддерживаются JPG, PNG, BMP, TIFF"
            )
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert('RGB')
                    st.success(f"✅ Изображение загружено: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке изображения: {str(e)}")
        
        elif upload_option == "Использовать URL":
            url = st.text_input("Введите URL изображения:", placeholder="https://example.com/image.jpg")
            if url:
                try:
                    import requests
                    from io import BytesIO
                    
                    response = requests.get(url, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    st.success("✅ Изображение загружено по URL")
                except Exception as e:
                    st.error(f"❌ Ошибка при загрузке по URL: {str(e)}")
        
        else:  # Примеры изображений
            example_option = st.selectbox(
                "Выберите пример:",
                ["Хороший сварной шов", "Плохой сварной шов", "Тестовое изображение 1", "Тестовое изображение 2"]
            )
            
            # Здесь можно добавить пути к примерным изображениям
            # Для демонстрации используем заглушки
            st.info("ℹ️ В рабочем приложении здесь будут реальные примеры изображений")
            st.warning("⚠️ Функция примеров требует добавления реальных изображений в папку examples/")
    
    with col2:
        if image is not None:
            st.markdown('<h3 class="sub-header">👁️ Предпросмотр</h3>', unsafe_allow_html=True)
            
            # Отображаем изображение
            st.image(image, caption="Загруженное изображение", use_column_width=True)
            
            # Информация об изображении
            width, height = image.size
            st.caption(f"Размер: {width} × {height} пикселей | Формат: RGB")
    
    # Кнопка предсказания
    if image is not None and st.button("🚀 Анализировать изображение", type="primary"):
        with st.spinner("🔍 Анализ изображения..."):
            # Создаем трансформации
            transform = get_transforms(config['INPUT_SIZE'])
            
            # Делаем предсказание
            start_time = time.time()
            pred_class, confidence, all_probs = predict_image(
                model, image, transform, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                class_names=class_names
            )
            inference_time = time.time() - start_time
            
            # Результаты
            st.markdown("---")
            st.markdown('<h3 class="sub-header">📊 Результаты анализа</h3>', unsafe_allow_html=True)
            
            # Отображаем результат с соответствующим стилем
            if pred_class == "good_weld":
                st.markdown(f"""
                <div class="prediction-good">
                    <h3>✅ Результат: КАЧЕСТВЕННЫЙ СВАРНОЙ ШОВ</h3>
                    <p>Модель определила, что сварной шов соответствует стандартам качества.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-bad">
                    <h3>❌ Результат: НЕКАЧЕСТВЕННЫЙ СВАРНОЙ ШОВ</h3>
                    <p>Модель обнаружила возможные дефекты сварного шва.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Уверенность модели
            st.markdown(f"**Уверенность модели:** {confidence:.2%}")
            
            # Визуализация уверенности
            confidence_percent = int(confidence * 100)
            fill_color = "#4CAF50" if pred_class == "good_weld" else "#F44336"
            
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {fill_color};">
                    {confidence_percent}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Детальная информация
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Время анализа", f"{inference_time:.3f} сек")
            
            with col_b:
                st.metric("Размер модели", f"{sum(p.numel() for p in model.parameters()):,} параметров")
            
            # Визуализация вероятностей для всех классов
            st.markdown("**Распределение вероятностей:**")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(class_names, all_probs, color=['#F44336', '#4CAF50'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Вероятность')
            ax.set_title('Вероятности по классам')
            
            # Добавляем значения на столбцы
            for i, (bar, prob) in enumerate(zip(bars, all_probs)):
                ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.2%}', va='center')
            
            st.pyplot(fig)
            
            # Дополнительная информация
            with st.expander("📋 Технические детали"):
                st.write(f"**Класс:** {pred_class}")
                st.write(f"**Индекс класса:** {class_names.index(pred_class)}")
                st.write(f"**Все вероятности:**")
                for class_name, prob in zip(class_names, all_probs):
                    st.write(f"  - {class_name}: {prob:.4f}")
                
                st.write(f"**Устройство:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                st.write(f"**Размер входа:** {config['INPUT_SIZE']}x{config['INPUT_SIZE']}")
    
    # Раздел с инструкциями, если изображение не загружено
    elif image is None:
        st.markdown("---")
        st.markdown("### 📋 Как использовать")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**1. Загрузите**\n\nВыберите изображение сварного шва")
        
        with col2:
            st.info("**2. Проанализируйте**\n\nНажмите кнопку 'Анализировать'")
        
        with col3:
            st.info("**3. Получите результат**\n\nОцените качество сварки")
    
    # Футер
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Weld Quality Classifier v1.0 | Создано с помощью PyTorch Lightning & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()