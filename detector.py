import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

def cargar_modelo(model_path: str):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    return processor, model


def clasificar_imagen(image_path: str, processor, model):
    if not os.path.isfile(image_path):
        print(f"⚠️  No se encontró el archivo: {image_path}\n")
        return None, None

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️  No se pudo abrir {image_path}: {e}\n")
        return None, None

    # Preprocesar
    inputs = processor(images=image, return_tensors="pt")

    # Inferencia
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    # Resultado
    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    print(f"Imagen: {image_path}")
    print(f"Predicción: {label}")
    print(f"Confianza: {confidence:.2%}")
    print("Probabilidades por clase:")
    for idx, prob in enumerate(probs):
        print(f"  {model.config.id2label[idx]}: {prob:.2%}")
    print()

    return label, confidence


def main(imgs: list[str], model_path: str):
    processor, model = cargar_modelo(model_path)
    for i, image in enumerate(imgs):
        if i>13:
            print("CAPTURAS DE PANTALLA (IMAGENES FALSAS)")
        clasificar_imagen(image, processor, model)


if __name__ == "__main__":
    MODEL_PATH = "./src/detect-model"

    images = [
        "./imagenes/bryan-fake.jpg",
        "./imagenes/bryan-fake-cara.jpg",
        "./imagenes/bryan-fake2.jpg",
        "./imagenes/bryan-fake2-cara.jpg",
        "./imagenes/simulacion2.jpg",
        "./imagenes/simulacion4-alta-calidad.png",
        "./imagenes/simulacion4-alta-calidad-cara.png",
        "./imagenes/bigote-falso.jpg",
        "./imagenes/barba-falsa.jpg",
        "./imagenes/bigote-falso-redimensionada.jpg",
        "./imagenes/barba-falsa-redimensionada.jpg",
        "./imagenes/brazo-raro.png",
        "./imagenes/tyler-modificado-redimensionada2.jpg",
        # CAPTURAS DE PANTALLA
        "./imagenes/capturas/bigote.png",
        "./imagenes/capturas/barba.png",
        "./imagenes/capturas/bryan1-cara.png",
        "./imagenes/capturas/bryan2-cara.png",
        "./imagenes/capturas/tyler.png",
    ]
    main(images, MODEL_PATH)