
# Fruit Freshness Detection (Apple, Banana, Orange)

<img width="1920" height="1020" alt="Screenshot 2026-01-24 133508" src="https://github.com/user-attachments/assets/04b2d6f4-f608-4cfc-8ffb-64ca500593f4" />
<img width="1920" height="1020" alt="Screenshot 2026-01-24 133807" src="https://github.com/user-attachments/assets/537c312d-e6b4-49c5-8ac6-7fbe062a1496" />

# Fruit Freshness Detection

Computer vision model that classifies fruits as **Fresh** or **Rotten** in real time.
Built with MobileNetV2 and deployed as a Flask web app.

🔗 **[Live App](https://fruit-freshness-production.up.railway.app)**

---

## Supported Fruits
Apple · Banana · Orange

---

## Pipeline

```
Fruit Image Upload
      │
      ▼
Image Preprocessing (resize + normalize)
      │
      ▼
MobileNetV2 Classification
      │
      ▼
Fresh / Rotten → Displayed in UI
```

---

## Why MobileNetV2?
- Lightweight architecture — fast inference
- Strong performance on limited datasets
- Mobile-friendly — suitable for real-world edge deployment

---

## Real-World Use Cases
- Fruit shops — quick quality checks at point of sale
- Warehouses — reduce wastage before storage
- Food supply chains — automated inspection on conveyor belts

---

## Tech Stack
Python · TensorFlow/Keras · MobileNetV2 · Flask · HTML

---

## Project Structure
```
├── app.py              # Flask web app
├── model.h5            # Trained MobileNetV2 model
├── requirements.txt
├── templates/          # HTML frontend
└── static/             # CSS and assets
```
```

---


