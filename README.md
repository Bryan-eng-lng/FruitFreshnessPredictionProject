
# Fruit Freshness Detection (Apple, Banana, Orange)

<img width="1920" height="1020" alt="Screenshot 2026-01-24 133508" src="https://github.com/user-attachments/assets/04b2d6f4-f608-4cfc-8ffb-64ca500593f4" />
<img width="1920" height="1020" alt="Screenshot 2026-01-24 133807" src="https://github.com/user-attachments/assets/537c312d-e6b4-49c5-8ac6-7fbe062a1496" />

This project detects whether a fruit is **Fresh** or **Rotten** using deep learning.  
It is designed as a practical AI solution for real-world fruit quality inspection in shops, warehouses, and supply chains.

Manual fruit inspection is slow, inconsistent, and often leads to wastage or poor-quality fruits reaching customers.  
This project explores how computer vision can automate and improve this process.

---

## Problem Statement

In real-world environments like fruit shops and warehouses:
- Quality checking is done manually
- Human judgment varies
- Rotten fruits sometimes pass inspection
- Good fruits may get wasted

The goal of this project is to build a **simple, fast, and usable AI system** that can classify fruit freshness from images.

---

## Project Scope

Currently supported fruits:
- Apple
- Banana
- Orange

Prediction classes:
- Fresh
- Rotten

---
## Systm Flow 
-Fruit Image → Deep Learning Model → Fresh / Rotten


---

## Model & Approach

- Used **MobileNetV2** for image classification
- Chosen for:
  - Lightweight architecture
  - Fast inference
  - Good performance on limited datasets

The model learns visual patterns such as:
- Color changes
- Texture variations
- Dark or spoiled regions

Images were resized and normalized before training.

---

## Web Application

A simple **Flask-based web application** was built to make the system usable for non-technical users.

Flow:
1. User uploads a fruit image
2. Clicks "Predict"
3. System displays Fresh or Rotten

This simulates how such a system could be used in real environments.

---

## Real-World Use Cases

- Fruit shops for quick quality checks
- Warehouses to reduce wastage
- Factories with conveyor belts and camera-based inspection
- Entry-level automation in food supply chains

In future, this system can be extended with:
- Sound alerts
- Automated rejection mechanisms
- Camera-based real-time detection

---

## Limitations & Real-World Learning

One key learning from this project is that **real-world data is rarely binary**.

In reality:
- Fruits may be partially rotten
- Some have only small spoiled areas
- Freshness exists on a spectrum, not just Fresh or Rotten

The current dataset mostly contains **clear cases** (fully fresh or fully rotten).  
Because of this, the model may struggle when:
- Only small rotten spots are present
- The fruit is partially spoiled

---

## Future Improvements

- Add partially rotten fruit images
- Introduce multiple spoilage levels (e.g., 60% fresh / 40% rotten)
- Improve dataset diversity (lighting, angles, backgrounds)
- Extend to more fruit categories

---

## Key Learnings

- Dataset quality matters more than model complexity
- Real-world ML problems are not clean or binary
- Building usable systems is as important as model accuracy
- Practical constraints shape better engineering decisions

---

## Conclusion

This project focuses on **practical learning and real-world applicability**, not just accuracy metrics.  
It helped me understand how AI systems behave outside ideal datasets and how to think beyond textbook problems.



