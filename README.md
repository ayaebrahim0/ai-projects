# ai-projects

A collection of AI / ML / DL / NLP / CV projects — models, notebooks, and experiments. Each project lives in its own folder with its own README, code, and results.

## Projects

| Project | Description | Stack |
|---|---|---|
| [StudyMate — Local AI Student Assistant](./studymate-ai-student-assistant) | Local (no external API) study assistant: custom-trained YOLOv8 layout detection + Tesseract OCR feeds a RAG chatbot, quiz generator, and mind-map builder, served through a bilingual (EN/AR) Gradio dashboard. | YOLOv8, Tesseract OCR, BART, ChromaDB, SentenceTransformers, Qwen2.5-1.5B, spaCy, Gradio |
| [Teeth Segmentation with U-Net](./teeth-segmentation-unet) | Binary segmentation model that predicts a pixel-level teeth mask from a photo, trained on a self-collected, hand-annotated dataset (291 images). U-Net decoder on a frozen, ImageNet-pretrained ResNet50 encoder, trained with Dice loss. | TensorFlow/Keras, ResNet50, U-Net, OpenCV |

More projects are added here over time — see each folder's README for details, setup instructions, and results.
