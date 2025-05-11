# CIFAR-10 Image Classifier

A modern web application that uses deep learning to classify images into 10 different categories using the CIFAR-10 dataset. Built with Flask and TensorFlow, featuring a beautiful glassmorphism UI design.

## Features

- 🖼️ Image Classification (10 categories)
- 🎯 Top 3 Predictions with confidence scores
- 🌓 Dark/Light Mode Toggle
- 🎨 Modern Glassmorphism UI
- 📱 Responsive Design
- 🔄 Real-time Image Preview
- 📊 Visual Confidence Bars

## Categories

- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐎 Horse
- 🚢 Ship
- 🚚 Truck

## Tech Stack

- **Backend**:
  - Python 3.8+
  - Flask (Web Framework)
  - TensorFlow/Keras (Deep Learning)
  - CIFAR-10 CNN Model

- **Frontend**:
  - HTML5
  - CSS3 (Custom Glassmorphism Design)
  - Vanilla JavaScript
  - Bootstrap 5 (UI Components)

## Quick Start

### Local Development

1. Clone and setup:
```bash
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open `http://localhost:5000` in your browser

### Vercel Deployment

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy to Vercel:
```bash
vercel
```

4. For production deployment:
```bash
vercel --prod
```

The app will be available at `https://your-app-name.vercel.app`

## Project Structure

```
cifar10-classifier/
├── app.py              # Flask application
├── templates/          # HTML templates
├── static/            # Static files
├── requirements.txt   # Dependencies
├── vercel.json        # Vercel configuration
└── README.md         # Documentation
```

## Usage

1. Upload an image
2. View real-time preview
3. Get top 3 predictions with confidence scores
4. Toggle dark/light mode as needed

## License

MIT License

## Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)
Project Link: [https://github.com/yourusername/cifar10-classifier](https://github.com/yourusername/cifar10-classifier) 