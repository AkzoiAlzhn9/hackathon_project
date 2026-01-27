# 🚀 AI Video Tutor — Платформа интерактивного обучения

Инновационное расширение для браузера, которое превращает обычный просмотр учебных видео в полноценный интерактивный урок с проверкой знаний в реальном времени.

## 🌟 Основные возможности
- **Automatic ASR (Whisper):** Локальное распознавание речи прямо из видеопотока.
- **Smart Segmentation:** Автоматическое деление видео на логические блоки на основе анализа текста.
- **AI Quiz Generation (RAKE):** Генерация проверочных тестов на лету по ключевым терминам из видео.
- **Learning Lock:** Блокировка дальнейшего просмотра, пока пользователь не ответит правильно на вопрос.
- **Progress Dashboard:** Наглядная панель со списком глав и статусом их прохождения.

## 🛠 Технологический стек
- **Backend:** FastAPI, Faster-Whisper, RAKE-NLTK, MoviePy.
- **Frontend (Extension):** JavaScript (Content Scripts, Popup API), CSS3 (Blur & Glassmorphism UI).
- **Processing:** NLP-анализ текста и автоматическая генерация дистракторов для тестов.

## 🚀 Быстрый запуск

### 1. Сервер (Backend)
```bash
pip install fastapi uvicorn faster-whisper rake-nltk moviepy
python main.py
http://127.0.0.1:8000/docs