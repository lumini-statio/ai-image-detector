import os
import json
import time
from datetime import datetime
from collections import defaultdict
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Confirmado con el modelo: {0: 'artificial', 1: 'real'}
LABELS_IA = {"artificial", "fake", "ai", "ai-generated", "ia"}
LABELS_REAL = {"real", "human", "authentic"}

CHECKPOINT_CADA = 5000
LOG_CADA = 500


def cargar_modelo(model_path: str):
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()
    return processor, model


def listar_imagenes_recursivo(directorio: str) -> list[str]:
    imagenes = []
    for root, _dirs, files in os.walk(directorio):
        for fname in files:
            if fname.lower().endswith(EXTENSIONES_VALIDAS):
                imagenes.append(os.path.join(root, fname))
    return imagenes


def clasificar_imagen(image_path: str, processor, model):
    if not os.path.isfile(image_path):
        return None, None, "archivo_no_encontrado"

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None, None, f"error_apertura: {e}"

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs))
    label = model.config.id2label[pred_id]
    confidence = float(probs[pred_id])

    return label, confidence, None


def categorizar_label(label: str) -> str:
    if label is None:
        return "error"
    normalizado = label.strip().lower()
    if normalizado in LABELS_IA:
        return "ia"
    if normalizado in LABELS_REAL:
        return "real"
    return "desconocido"


def guardar_resultados(output_path: str, stats: dict, detalle: list, parcial: bool = False):
    data = {
        "timestamp": datetime.now().isoformat(),
        "parcial": parcial,
        "resumen": stats,
        "detalle": detalle,
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


def calcular_resumen(contadores: dict, total_procesadas: int, total_encontradas: int) -> dict:
    total_ia = contadores["ia"]
    total_real = contadores["real"]
    total_desconocido = contadores["desconocido"]
    total_error = contadores["error"]

    def pct(n):
        return round((n / total_procesadas) * 100, 2) if total_procesadas else 0.0

    return {
        "total_imagenes_encontradas": total_encontradas,
        "total_imagenes_procesadas": total_procesadas,
        "clasificadas_como_ia": total_ia,
        "clasificadas_como_reales": total_real,
        "clasificadas_desconocido": total_desconocido,
        "errores": total_error,
        "porcentaje_ia": pct(total_ia),
        "porcentaje_real": pct(total_real),
        "porcentaje_desconocido": pct(total_desconocido),
        "porcentaje_error": pct(total_error),
    }


def main(dataset_dir: str, model_path: str, output_json: str, guardar_detalle: bool = True):
    processor, model = cargar_modelo(model_path)

    print(f"Labels del modelo: {model.config.id2label}\n")

    imagenes = listar_imagenes_recursivo(dataset_dir)
    total_encontradas = len(imagenes)
    print(f"Se encontraron {total_encontradas} imágenes en '{dataset_dir}'\n")

    contadores = defaultdict(int)
    detalle = []
    inicio = time.time()

    for i, image_path in enumerate(imagenes, start=1):
        label, confidence, error = clasificar_imagen(image_path, processor, model)

        if error is not None:
            categoria = "error"
        else:
            categoria = categorizar_label(label)

        contadores[categoria] += 1

        if guardar_detalle:
            detalle.append({
                "ruta": image_path,
                "label_crudo": label,
                "categoria": categoria,
                "confianza": round(confidence, 4) if confidence is not None else None,
            })

        if i % LOG_CADA == 0 or i == total_encontradas:
            transcurrido = time.time() - inicio
            velocidad = i / transcurrido if transcurrido > 0 else 0
            restante = (total_encontradas - i) / velocidad if velocidad > 0 else 0
            print(f"[{i}/{total_encontradas}] IA={contadores['ia']} Real={contadores['real']} "
                  f"Desconocido={contadores['desconocido']} Error={contadores['error']} "
                  f"| {velocidad:.1f} img/s | ETA: {restante/60:.1f} min")

        if i % CHECKPOINT_CADA == 0:
            stats_parcial = calcular_resumen(contadores, i, total_encontradas)
            guardar_resultados(output_json, stats_parcial, detalle, parcial=True)
            print(f"Checkpoint guardado en '{output_json}'\n")
            
            if i < total_encontradas:
                time.sleep(5)

    stats_final = calcular_resumen(contadores, len(imagenes), total_encontradas)
    guardar_resultados(output_json, stats_final, detalle, parcial=False)

    print("\n=== RESUMEN FINAL ===")
    for k, v in stats_final.items():
        print(f"{k}: {v}")
    print(f"\nResultados exportados a: {output_json}")


if __name__ == "__main__":
    MODEL_PATH = "./src/detect-model"
    DATASET_DIR = "/data/megaface"
    OUTPUT_JSON = "./resultados_analisis.json"

    main(DATASET_DIR, model_path=MODEL_PATH, output_json=OUTPUT_JSON, guardar_detalle=True)