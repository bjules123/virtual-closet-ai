# AI Virtual Closet

The **AI Virtual Closet** is a computer vision powered application that helps users organize their clothing digitally and receive outfit recommendations.

Users can upload images of their clothing items, and the system automatically analyzes them using machine learning to identify clothing attributes such as color, pattern, and sleeve type. The application then allows users to search their wardrobe and generate outfit suggestions based on events or daily activities.

The goal of this project is to demonstrate how **AI and computer vision can simplify everyday decision making**, such as choosing what to wear.

---

## Features

• Upload images of clothing items
• Automatic clothing classification using AI
• Detect attributes such as:

* clothing type
* color
* sleeve length
* pattern

• Filter clothing by category
• Generate outfit recommendations
• Suggest outfits based on events or occasions
• Optional weather-based outfit suggestions

---

## AI / Machine Learning

This project uses computer vision models to analyze clothing images.

### Model

* Vision Transformer (ViT)
* Trained on a clothing dataset with **15 clothing categories**

### Performance

* Model accuracy: **78.5% classification accuracy**

### Dataset

Clothing dataset containing categories such as:

* shirts
* dresses
* pants
* jackets
* sweaters

---

## Technology Stack

Frontend

* React (planned mobile interface)
* HTML / CSS

Backend

* Python
* FastAPI

Machine Learning

* PyTorch
* Hugging Face Transformers
* Vision Transformer (ViT)

Computer Vision

* YOLO / image classification models

---

## How It Works

1. User uploads an image of a clothing item
2. The AI model analyzes the image
3. Clothing attributes are extracted
4. Item is added to the user's digital closet
5. The recommendation engine suggests outfits

---

## Example Workflow

Upload clothing → AI detects attributes → Item stored in closet → Outfit recommendation generated

---

## Future Improvements

Planned extensions include:

• Weather-based outfit recommendations
• Integration with shopping APIs for clothing suggestions
• Mobile app version
• Automatic background removal for clothing images
• Personalized style learning

---

## Author

Brianna Jules
