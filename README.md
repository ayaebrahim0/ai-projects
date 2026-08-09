# ai-projects

A collection of AI / ML / DL / NLP / CV projects — models, notebooks, and experiments. Each project lives in its own folder with its own README, code, and results.

## Projects

| Project | Description | Stack |
|---|---|---|
| [StudyMate — Local AI Student Assistant](./studymate-ai-student-assistant) | Local (no external API) study assistant: custom-trained YOLOv8 layout detection + Tesseract OCR feeds a RAG chatbot, quiz generator, and mind-map builder, served through a bilingual (EN/AR) Gradio dashboard. | YOLOv8, Tesseract OCR, BART, ChromaDB, SentenceTransformers, Qwen2.5-1.5B, spaCy, Gradio |
| [Teeth Segmentation with U-Net](./teeth-segmentation-unet) | Binary segmentation model that predicts a pixel-level teeth mask from a photo, trained on a self-collected, hand-annotated dataset (291 images). U-Net decoder on a frozen, ImageNet-pretrained ResNet50 encoder, trained with Dice loss. | TensorFlow/Keras, ResNet50, U-Net, OpenCV |
| [IPL 2023 Auction — EDA](./ipl-2023-auction-eda) | Exploratory analysis of the IPL 2023 player auction: data cleaning plus visual answers to team-composition and pricing questions (base price by role, players per team, top-priced players). | Pandas, Seaborn, Matplotlib |
| [COVID-19 Global Dataset Analysis](./covid19-global-analysis) | Exploratory analysis of global COVID-19 daily and summary data: mortality/fatality rates by country and continent, top-affected countries, choropleth maps, and per-country case/death time series. | Pandas, Plotly, Seaborn, Matplotlib |
| [Fuel Efficiency (MPG) Regression](./ann-regression-auto-mpg) | Feed-forward neural network predicting a car's MPG from its specs, with z-score normalization and early stopping / LR reduction / best-checkpoint callbacks. | TensorFlow/Keras, Scikit-learn |
| [Driver Drowsiness Detection](./driver-drowsiness-detection-cnn) | CNN classifying eye state (Open/Closed) as the core signal for a driver-drowsiness alert system. ~99.9% test accuracy. | TensorFlow/Keras, OpenCV |
| [Arabic Twitter Sentiment Analysis](./arabic-sentiment-lstm) | Binary sentiment classification of Arabic tweets — compares a TF-IDF + Logistic Regression baseline against a Bidirectional LSTM, including a documented failed attempt with a plain LSTM. | TensorFlow/Keras, Scikit-learn, NLP |

More projects are added here over time — see each folder's README for details, setup instructions, and results.
