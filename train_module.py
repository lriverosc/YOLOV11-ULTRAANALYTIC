from __future__ import annotations
import sys, os, math, shutil, traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from ultralytics import YOLO

# ====== CONFIG BÁSICA ======
IMGSZ = 640
CONF_PSEUDO = 0.30   # conf. para pseudo-etiquetado
EPOCHS = 50
BATCH = 8
VAL_SPLIT = 0.20
# ===========================

CLASSES = ["vehiculo_menor","buses_urbanos","motos","peatones","vehiculos_pesados"]

def device_string():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def ensure_dirs(base: Path):
    (base/"images/train").mkdir(parents=True, exist_ok=True)
    (base/"images/val").mkdir(parents=True, exist_ok=True)
    (base/"labels/train").mkdir(parents=True, exist_ok=True)
    (base/"labels/val").mkdir(parents=True, exist_ok=True)

def write_dataset_yaml(base: Path):
    yaml = f"""path: {base.as_posix()}
train: images/train
val: images/val
names:
  0: vehiculo_menor
  1: buses_urbanos
  2: motos
  3: peatones
  4: vehiculos_pesados
"""
    (base/"dataset.yaml").write_text(yaml, encoding="utf-8")

def yolo_line(xyxy, cls_id, imgw, imgh):
    x1,y1,x2,y2 = xyxy
    w = max(1.0, imgw); h = max(1.0, imgh)
    cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
    bw = (x2-x1)/w; bh = (y2-y1)/h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

def assign_split(idx, total, val_split=VAL_SPLIT):
    return "val" if (idx / max(1,total)) < val_split else "train"

class TrainerUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Constructor de Dataset YOLO + Entrenamiento (GPU)")
        self.resize(920, 560)

        self.video_path: Path|None = None
        self.out_dir: Path|None = None

        # UI
        lay = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.le_video = QtWidgets.QLineEdit(); self.le_video.setPlaceholderText("Video .mp4")
        self.le_out   = QtWidgets.QLineEdit(); self.le_out.setPlaceholderText("Carpeta dataset")

        btn_v = QtWidgets.QPushButton("Buscar video…")
        btn_o = QtWidgets.QPushButton("Elegir carpeta…")
        btn_v.clicked.connect(self.pick_video)
        btn_o.clicked.connect(self.pick_out)

        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.le_video); row1.addWidget(btn_v)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.le_out);   row2.addWidget(btn_o)

        form.addRow("Video .mp4:", row1)
        form.addRow("Salida dataset:", row2)

        self.sb_stride = QtWidgets.QDoubleSpinBox(); self.sb_stride.setDecimals(2)
        self.sb_stride.setRange(0.03, 2.0); self.sb_stride.setSingleStep(0.05); self.sb_stride.setValue(0.5)
        form.addRow("Cada X segundos:", self.sb_stride)

        self.sb_epochs = QtWidgets.QSpinBox(); self.sb_epochs.setRange(1, 300); self.sb_epochs.setValue(EPOCHS)
        form.addRow("epochs:", self.sb_epochs)

        self.sb_batch = QtWidgets.QSpinBox(); self.sb_batch.setRange(1, 64); self.sb_batch.setValue(BATCH)
        form.addRow("batch:", self.sb_batch)

        lay.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Crear dataset y entrenar")
        self.btn_run.setStyleSheet("background:#28A745;color:white;")
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.setStyleSheet("background:#DC3545;color:white;")
        btns.addWidget(self.btn_run); btns.addWidget(self.btn_cancel)
        lay.addLayout(btns)

        self.pbar = QtWidgets.QProgressBar(); self.pbar.setValue(0)
        lay.addWidget(self.pbar)

        self.log = QtWidgets.QPlainTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)

        self.btn_run.clicked.connect(self.run_all)
        self.btn_cancel.clicked.connect(lambda: QtWidgets.QApplication.quit())

    # -------- helpers --------
    def pick_video(self):
        p,_=QtWidgets.QFileDialog.getOpenFileName(self,"Video", "", "Video (*.mp4 *.mov *.avi)")
        if p: self.video_path=Path(p); self.le_video.setText(p)

    def pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self,"Carpeta de salida")
        if d: self.out_dir=Path(d); self.le_out.setText(d)

    def logln(self, s): 
        self.log.appendPlainText(s); print(s, flush=True)

    # -------- pipeline --------
    def run_all(self):
        try:
            if not self.video_path: 
                self.logln("Elige un video."); return
            if not self.out_dir:
                self.logln("Elige carpeta de salida."); return

            ds_dir = self.out_dir
            ensure_dirs(ds_dir)
            write_dataset_yaml(ds_dir)

            # 1) Extraer frames
            stride_s = float(self.sb_stride.value())
            self.extract_frames(self.video_path, ds_dir, stride_s)

            # 2) Pseudo-etiquetar (arranque) con YOLOv8n (COCO) → mapea a tus clases
            self.pseudo_label(ds_dir)

            # 3) Entrenar
            self.train(ds_dir, epochs=int(self.sb_epochs.value()), batch=int(self.sb_batch.value()))

            QtWidgets.QMessageBox.information(self, "Listo", "Entrenamiento finalizado.")
        except Exception as e:
            tb = traceback.format_exc()
            self.logln(tb)
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def extract_frames(self, video: Path, ds_dir: Path, stride_seconds: float):
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"No pude abrir el video: {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        every = max(1, int(round(fps*stride_seconds)))
        self.logln(f"FPS detectados: {fps:.2f} | extracción cada {stride_seconds}s (~cada {every} frames)")
        idx = 0; saved = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if idx % every == 0:
                split = assign_split(idx, total)
                out_img = ds_dir / f"images/{split}/{video.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out_img), frame)
                saved += 1
                if saved % 25 == 0:
                    self.pbar.setValue(min(90, int(90*idx/max(1,total))))
            idx += 1
        cap.release()
        self.logln(f"Frames guardados: {saved}")

    def pseudo_label(self, ds_dir: Path):
        self.logln("Cargando modelo base para pseudo-etiquetar…")
        model = YOLO("yolov8n.pt")
        model.to(device_string())

        for split in ["train","val"]:
            img_dir = ds_dir / f"images/{split}"
            lbl_dir = ds_dir / f"labels/{split}"
            lbl_dir.mkdir(parents=True, exist_ok=True)
            imgs = sorted(img_dir.glob("*.jpg"))
            n = len(imgs)
            for i, imgp in enumerate(imgs, 1):
                im = cv2.imread(str(imgp))
                h, w = im.shape[:2]
                res = model.predict(source=[im], imgsz=IMGSZ, conf=CONF_PSEUDO, verbose=False)[0]
                lines: List[str] = []
                if res.boxes is not None and len(res.boxes) > 0:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    clss = res.boxes.cls.cpu().numpy().astype(int)
                    names = res.names
                    for bb, c in zip(xyxy, clss):
                        cname = names.get(int(c), str(c)).lower()
                        # mapear COCO -> nuestras 5 clases
                        if cname in ("car","truck","bus","motorcycle","person"):
                            if cname=="car": cid=0
                            elif cname=="bus": cid=1
                            elif cname=="motorcycle": cid=2
                            elif cname=="person": cid=3
                            else: cid=4  # truck
                            x1,y1,x2,y2 = [float(v) for v in bb.tolist()]
                            lines.append(yolo_line((x1,y1,x2,y2), cid, w, h))
                (lbl_dir/f"{imgp.stem}.txt").write_text("".join(lines), encoding="utf-8")
                if i % 20 == 0:
                    self.pbar.setValue(min(95, int(90 + 5*i/max(1,n))))
        self.logln("Pseudo-etiquetado inicial listo.")

    def train(self, ds_dir: Path, epochs: int, batch: int):
        self.logln("Iniciando entrenamiento…")
        args = dict(
            model="yolov8n.pt",
            data=str(ds_dir/"dataset.yaml"),
            epochs=epochs,
            imgsz=IMGSZ,
            batch=batch,
            device=0 if device_string().startswith("cuda") else "cpu",
            workers=2,
            patience=5,
            name="train",
            deterministic=True,
            seed=42,
            pretrained=True
        )
        self.logln(str(args))
        model = YOLO(args["model"])
        model.train(**args)
        self.logln("Entrenamiento finalizado. Pesos en runs/detect/train/weights/best.pt")
        self.pbar.setValue(100)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = TrainerUI()
    ui.show()
    sys.exit(app.exec())
