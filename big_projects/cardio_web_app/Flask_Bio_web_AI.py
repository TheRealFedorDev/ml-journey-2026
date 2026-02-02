# app.py - ПОЛНЫЙ FLASK ПРИЛОЖЕНИЕ
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# ПРОВЕРКА: есть ли файлы модели
print("🔍 Проверяем наличие файлов модели...")
model_files = ['best_cardio_model.pkl', 'cardio_preprocessor.pkl']
missing_files = [f for f in model_files if not os.path.exists(f)]

if missing_files:
    print(f"❌ Отсутствуют файлы: {missing_files}")
    print("   Сначала обучи модель с помощью cardio_sklearn_only.py")
    exit()
else:
    print("✅ Все файлы найдены")

# ЗАГРУЗКА МОДЕЛИ И ПРЕПРОЦЕССОРА
print("📦 Загружаем модель и препроцессор...")
try:
    model = joblib.load('best_cardio_model.pkl')
    preprocessor = joblib.load('cardio_preprocessor.pkl')
    print("✅ Модель и препроцессор загружены успешно!")
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    exit()


# ГЛАВНАЯ СТРАНИЦА С ФОРМОЙ
@app.route('/')
def home():
    """Отображает главную страницу с формой"""
    return render_template('index.html')


# API ДЛЯ ПРЕДСКАЗАНИЙ
@app.route('/predict', methods=['POST'])
def predict():
    """Обрабатывает данные из формы и возвращает предсказание"""
    try:
        print("📥 Получен запрос на предсказание...")

        # Получаем данные из формы
        data = {
            'age': int(request.form['age']),
            'gender': int(request.form['gender']),
            'height': int(request.form['height']),
            'weight': float(request.form['weight']),
            'ap_hi': int(request.form['ap_hi']),
            'ap_lo': int(request.form['ap_lo']),
            'cholesterol': int(request.form['cholesterol']),
            'gluc': int(request.form['gluc']),
            'smoke': int(request.form.get('smoke', 0)),
            'alco': int(request.form.get('alco', 0)),
            'active': int(request.form.get('active', 0))
        }

        print(f"   Данные пациента: {data}")

        # Создаём DataFrame
        df = pd.DataFrame([data])

        # Добавляем вычисляемые признаки (как при обучении)
        df['age_years'] = df['age'] // 365
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

        print(f"   Вычисленные признаки: age_years={df['age_years'].iloc[0]}, bmi={df['bmi'].iloc[0]:.1f}")

        # Преобразуем через препроцессор
        X = preprocessor.transform(df)

        # Предсказываем вероятность
        probability = model.predict_proba(X)[0][1]

        # Определяем уровень риска
        if probability < 0.3:
            risk_level = "ОЧЕНЬ НИЗКИЙ"
            color = "green"
            emoji = "✅"
        elif probability < 0.5:
            risk_level = "НИЗКИЙ"
            color = "lightgreen"
            emoji = "👍"
        elif probability < 0.7:
            risk_level = "УМЕРЕННЫЙ"
            color = "orange"
            emoji = "⚠️"
        elif probability < 0.9:
            risk_level = "ВЫСОКИЙ"
            color = "red"
            emoji = "❗"
        else:
            risk_level = "ОЧЕНЬ ВЫСОКИЙ"
            color = "darkred"
            emoji = "🚨"

        # Генерируем рекомендации
        recommendations = generate_recommendations(data, probability)

        # Формируем результат
        result = {
            'success': True,
            'probability': float(probability),
            'percentage': f"{probability * 100:.1f}%",
            'risk_level': risk_level,
            'color': color,
            'emoji': emoji,
            'recommendations': recommendations,
            'patient_data': data
        }

        print(f"   Результат: вероятность {probability * 100:.1f}%, риск: {risk_level}")

        # Отображаем результат в HTML
        return render_template('index.html', result=result, form_data=data)

    except Exception as e:
        print(f"Ошибка: {e}")
        return render_template('index.html',
                               error=f"Ошибка обработки: {str(e)}")


# ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ
def generate_recommendations(patient_data, probability):
    """Генерирует персонализированные рекомендации"""
    recommendations = []

    # Возраст
    age_years = patient_data['age'] // 365
    if age_years > 50:
        recommendations.append("Регулярные обследования у кардиолога (раз в год)")

    # Давление
    if patient_data['ap_hi'] >= 140 or patient_data['ap_lo'] >= 90:
        recommendations.append("Контроль артериального давления ежедневно")
        recommendations.append("Ограничить потребление соли")

    # Холестерин
    if patient_data['cholesterol'] >= 2:
        recommendations.append("Проверить липидный профиль крови")
        recommendations.append("Уменьшить потребление жирной пищи")

    # Курение
    if patient_data['smoke'] == 1:
        recommendations.append("Рекомендуется бросить курить")

    # Вес
    bmi = patient_data['weight'] / ((patient_data['height'] / 100) ** 2)
    if bmi >= 25:
        recommendations.append("Контроль веса, умеренные физические нагрузки")

    # Активность
    if patient_data['active'] == 0:
        recommendations.append("Добавить физическую активность (30 мин/день)")

    # Общие рекомендации по уровню риска
    if probability > 0.7:
        recommendations.append("Срочная консультация кардиолога + ЭКГ")
    elif probability > 0.5:
        recommendations.append("Рекомендуется консультация кардиолога")
    else:
        recommendations.append("Продолжайте здоровый образ жизни!")

    return recommendations[:6]  # Максимум 6 рекомендаций


# API ДЛЯ МОБИЛЬНЫХ ПРИЛОЖЕНИЙ (JSON)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint для мобильных приложений (возвращает JSON)"""
    try:
        data = request.json

        df = pd.DataFrame([data])
        df['age_years'] = df['age'] // 365
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

        X = preprocessor.transform(df)
        probability = model.predict_proba(X)[0][1]

        return jsonify({
            'success': True,
            'probability': float(probability),
            'risk': 'high' if probability > 0.5 else 'low'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ЗАПУСК СЕРВЕРА
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("СЕРВЕР КАРДИОЛОГИЧЕСКОЙ ДИАГНОСТИКИ ЗАПУЩЕН!")
    print("=" * 50)
    print("Открой в браузере: http://127.0.0.1:5000")
    print("Для остановки нажми Ctrl+C")
    print("=" * 50)

    app.run(debug=True, host='127.0.0.1', port=5000)