###  Telco Customer Churn Prediction ⭐
**[GitHub](https://github.com/skelinge/churn-prediction)** | LightGBM, Optuna, SHAP

Предсказание оттока клиентов телекоммуникационной компании (7K записей, дисбаланс 27%, реальные данные IBM).

**Ключевые результаты:**
| Модель | PR-AUC ↑ | ROC-AUC | F1-score |
|--------|----------|---------|----------|
| Logistic Regression | ~0.58 | ~0.81 | ~0.59 |
| Random Forest | 0.6067 | 0.8234 | 0.6085 |
| **LightGBM + Optuna** | **0.6524** | **0.8403** | **0.6245** |

**Технические решения:**
- Feature engineering: `ChargesPerMonth`, `IsNewCustomer` (первые 6 месяцев — пик churn)
- Предобработка: RobustScaler для числовых признаков, LabelEncoding + OneHotEncoding для категориальных
- Оптимизация: Optuna (50 trials) с прямой оптимизацией по PR-AUC
- Интерпретируемость: SHAP выявил ключевые драйверы — `Contract_Two_year`, `tenure`, `InternetService_Fiber optic`

**Стек:** Python, LightGBM, Optuna, SHAP, Scikit-learn, Pandas, Seaborn

| Модель | PR-AUC | ROC-AUC | F1 |
|--------|--------|---------|-----|
| Logistic Regression | ~0.58 | ~0.81 | ~0.59 |
| Random Forest | 0.6067 | 0.8234 | 0.6085 |
| **LightGBM + Optuna** | **0.6524** | **0.8403** | **0.6245** |

<img width="1200" height="1125" alt="shap_summary" src="https://github.com/user-attachments/assets/7b31619e-336c-43d6-9eb0-9ddb0aaaafc8" />
<img width="960" height="720" alt="pr_curves" src="https://github.com/user-attachments/assets/64af58a8-a382-4715-b0f1-61ab7206fef3" />
<img width="900" height="750" alt="confusion_matrix" src="https://github.com/user-attachments/assets/c6b06c8e-73cf-4f54-9c4d-2eb1e681a7c6" />
