# YOLOV11 - ULTRAANALYTIC 🚦📊

Sistema de **detección, conteo y análisis vehicular/peatonal** basado en **YOLOv8/YOLOv11**, con interfaz **PySide6 (Qt)** y soporte para **videos de dron**.  
Permite definir **regiones de interés (ROI)**, realizar **conteo por categorías**, trazar **direcciones de flujo** y **exportar resultados profesionales a Excel**.

---

## ✨ Características principales

- **Entrenamiento** personalizado desde videos (`train_module.py`).
- **Definición de ROI** con editor visual, guardado en `roi.json` o `<video>.roi.json`.
- **Detección en tiempo real** de:
  - 🚗 Vehículo menor (`car`)
  - 🚌 Buses urbanos (`bus`)
  - 🏍️ Motocicletas (`motorcycle`)
  - 🚶‍♂️ Peatones (`person`)
  - 🚙 Vehículos estacionados (categorías derivadas)
- **Conteo único por ID** → evita duplicados y estacionados.
- **Exportación a Excel** con:
  - Bordes y estilos profesionales
  - Columnas autoajustables
  - Conteo por categoría, fecha e intersección
- **Direcciones de tránsito** detectadas automáticamente:
  - Línea curva por carril
  - Flecha al final del tramo con el color del carril
- **UI Dark Mode** con botones animados (*Fizzy CSS Button*).
- **Loaders animados** (*Particle Orbit*) como splash y durante procesos.
- Compatible con **CUDA (NVIDIA RTX)** para aceleración en GPU.

---

## 📂 Estructura del proyecto

```
YOLOV11-ULTRAANALYTIC/
├── main_app.py          # Interfaz principal de detección y conteo
├── train_module.py      # Genera dataset desde videos y entrena modelos
├── requirements.txt     # Dependencias del proyecto
├── README.md            # Documentación principal
├── .gitignore
├── weights/             # Modelos entrenados (best.pt, last.pt)
├── datasets/            # Carpeta para datasets (no versionada en Git)
│   ├── images/train/
│   ├── images/val/
│   ├── labels/train/
│   └── labels/val/
├── runs/                # Resultados de entrenamiento (ignorada en git)
└── roi.json             # ROI por defecto (opcional)
```

---

## 🧠 Modelos de aprendizaje

### Clases detectadas
El sistema está entrenado para reconocer las siguientes clases:

- `car` → Vehículos menores
- `bus` → Buses urbanos
- `motorcycle` → Motocicletas
- `person` → Peatones
- `car_parked`, `bus_parked`, `motorcycle_parked` → Detectados como estacionados (via tracker temporal)

### Entrenamiento (`train_module.py`)
- Entrada: video `.mp4` (ej. desde dron).
- Salida: dataset YOLO (`images/`, `labels/`) + `dataset.yaml`.
- Modelo base: `yolov8n.pt` o `yolov8s.pt`.
- Parametrización:
  - Épocas: 20–50
  - Tamaño de imagen: 640
  - Batch ajustable según GPU
- Guardado automático en `runs/detect/train.../weights/`.

Ejemplo:
```bash
python train_module.py --video videos/interseccion.mp4 --out datasets/interseccion   --epochs 30 --imgsz 640
```

El mejor modelo se guarda como `best.pt` y se carga por defecto en la aplicación.

---

## 🖥️ Uso de la aplicación (`main_app.py`)

### 1) Cargar video
```bash
python main_app.py
```
- Selecciona el video `.mp4`.
- Se abrirá el **editor ROI** → dibuja el polígono de intersección → presiona `S` para guardar.
- Pregunta: “¿Deseas guardar este ROI?” → **Aceptar**.

### 2) Selección del modelo
- Automáticamente busca `runs/.../best.pt`.
- También puedes cargar manualmente otro `.pt`.

### 3) Iniciar análisis
- Botón **Iniciar** → cambia a **“Analizando datos…”**.
- Durante el proceso verás:
  - Progreso por frame (y tiempo estimado).
  - Etiquetas de detección más pequeñas (menos intrusivas).
  - Trazos de flujo + flechas curvas en cada calle.
- Al terminar: botón → **“Terminado”**.

### 4) Exportar resultados
- Botón **Exportar Excel** → crea un archivo con:
  - Fecha de análisis
  - Nombre de intersección
  - Conteo por categoría
  - Formato con bordes y autoajuste de columnas

---

## ⚙️ Requisitos

- Python 3.10 – 3.13
- GPU NVIDIA (CUDA 12.x recomendado)
- Dependencias:
  ```bash
  pip install -r requirements.txt
  # si tienes CUDA 12.4:
  pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
  ```

---

## 📝 Ejemplo de `roi.json`

```json
{
  "normalized": true,
  "invert": false,
  "polygons": [
    [[0.10, 0.35], [0.90, 0.35], [0.90, 0.65], [0.10, 0.65]]
  ]
}
```

- Coordenadas normalizadas (0–1).
- Puede contener varios polígonos en `"polygons"`.

---

## 📊 Ejemplo de salida Excel

| Categoría         | Conteo |
|-------------------|--------|
| Vehículos menores | 154    |
| Buses urbanos     | 12     |
| Motocicletas      | 38     |
| Peatones          | 89     |
| Estacionados      | 21     |

---

## 🔮 Futuras mejoras

- Refinar detección de peatones en escenas con alta altura de dron.  
- Integración con mapas GIS para geo-referenciar intersecciones.  
- Dashboards en tiempo real (streamlit/plotly).  
- Entrenamiento incremental con nuevos videos.  

---

## 📜 Licencia
Este proyecto está bajo la licencia MIT, ver archivo [LICENSE](LICENSE).
