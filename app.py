from __future__ import annotations

import ast
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "masked_face_model.onnx"

FALLBACK_CLASS_NAMES = {
    0: "with_mask",
    1: "without_mask",
    2: "mask_weared_incorrect",
}

CLASS_COLORS = {
    "with_mask": (46, 204, 113),
    "without_mask": (52, 73, 220),
    "mask_weared_incorrect": (0, 191, 255),
}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


class MaskDetector:
    """ONNX YOLO detector used by the Tkinter application."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

        self.model_path = model_path
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_height = self._dimension_or_default(input_meta.shape[-2], 640)
        self.input_width = self._dimension_or_default(input_meta.shape[-1], 640)

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.class_names = self._read_class_names(metadata.get("names"))

    @staticmethod
    def _dimension_or_default(value: object, default: int) -> int:
        return int(value) if isinstance(value, (int, np.integer)) else default

    @staticmethod
    def _read_class_names(raw_names: Optional[str]) -> dict[int, str]:
        if not raw_names:
            return FALLBACK_CLASS_NAMES.copy()

        try:
            parsed = ast.literal_eval(raw_names)
            if isinstance(parsed, dict):
                return {int(key): str(value) for key, value in parsed.items()}
            if isinstance(parsed, (list, tuple)):
                return {index: str(value) for index, value in enumerate(parsed)}
        except (ValueError, SyntaxError, TypeError):
            pass

        return FALLBACK_CLASS_NAMES.copy()

    def _preprocess(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        frame_height, frame_width = frame.shape[:2]
        scale = min(
            self.input_width / frame_width,
            self.input_height / frame_height,
        )
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))

        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        canvas = np.full(
            (self.input_height, self.input_width, 3),
            114,
            dtype=np.uint8,
        )
        pad_left = (self.input_width - resized_width) // 2
        pad_top = (self.input_height - resized_height) // 2
        canvas[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, pad_left, pad_top

    @staticmethod
    def _class_aware_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Apply greedy NMS independently for each class."""
        kept: list[int] = []

        for class_id in np.unique(class_ids):
            class_indexes = np.where(class_ids == class_id)[0]
            order = class_indexes[np.argsort(scores[class_indexes])[::-1]]

            while len(order) > 0:
                current = int(order[0])
                kept.append(current)
                if len(order) == 1:
                    break

                remaining = order[1:]
                current_box = boxes[current]
                other_boxes = boxes[remaining]

                left = np.maximum(current_box[0], other_boxes[:, 0])
                top = np.maximum(current_box[1], other_boxes[:, 1])
                right = np.minimum(current_box[2], other_boxes[:, 2])
                bottom = np.minimum(current_box[3], other_boxes[:, 3])

                intersection_width = np.maximum(0, right - left)
                intersection_height = np.maximum(0, bottom - top)
                intersection = intersection_width * intersection_height

                current_area = max(0.0, current_box[2] - current_box[0]) * max(
                    0.0, current_box[3] - current_box[1]
                )
                other_areas = np.maximum(0, other_boxes[:, 2] - other_boxes[:, 0]) * np.maximum(
                    0, other_boxes[:, 3] - other_boxes[:, 1]
                )
                union = current_area + other_areas - intersection
                iou = np.divide(
                    intersection,
                    union,
                    out=np.zeros_like(intersection),
                    where=union > 0,
                )
                order = remaining[iou <= iou_threshold]

        return sorted(kept, key=lambda index: float(scores[index]), reverse=True)

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
    ) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []

        tensor, scale, pad_left, pad_top = self._preprocess(frame)
        raw_output = self.session.run(None, {self.input_name: tensor})[0]
        predictions = np.asarray(raw_output)

        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise RuntimeError(f"Output model không hợp lệ: {predictions.shape}")

        # Ultralytics exports detections as [1, 4 + classes, 8400].
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise RuntimeError(f"Không đủ cột trong output model: {predictions.shape}")

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = np.max(class_scores, axis=1)

        valid = scores >= confidence_threshold
        boxes_xywh = boxes_xywh[valid]
        class_ids = class_ids[valid]
        scores = scores[valid]
        if len(scores) == 0:
            return []

        # Convert model-space xywh boxes to original-image xyxy boxes.
        boxes = np.empty_like(boxes_xywh, dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

        frame_height, frame_width = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height - 1)

        keep_indexes = self._class_aware_nms(
            boxes,
            scores,
            class_ids,
            iou_threshold,
        )

        detections: list[Detection] = []
        for index in keep_indexes:
            x1, y1, x2, y2 = boxes[index].round().astype(int).tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, f"class_{class_id}"),
                    confidence=float(scores[index]),
                    box=(x1, y1, x2, y2),
                )
            )

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
) -> tuple[np.ndarray, Counter[str]]:
    annotated = frame.copy()
    counts: Counter[str] = Counter()

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        counts[detection.class_name] += 1
        color = CLASS_COLORS.get(detection.class_name, (255, 170, 0))
        label = f"{detection.class_name} {detection.confidence:.0%}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_width + 8, y1),
            color,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height + 2, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, counts


class MaskDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Mask Detection - YOLO ONNX")
        self.root.geometry("1280x800")
        self.root.minsize(980, 650)

        self.detector: Optional[MaskDetector] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.after_id: Optional[str] = None
        self.stream_running = False
        self.source_kind = "Chưa có nguồn dữ liệu"
        self.source_path: Optional[Path] = None
        self.last_input_frame: Optional[np.ndarray] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        self.confidence_var = tk.DoubleVar(value=0.35)
        self.confidence_text = tk.StringVar(value="0.35")
        self.status_var = tk.StringVar(value="Đang khởi động...")
        self.source_var = tk.StringVar(value=self.source_kind)
        self.fps_var = tk.StringVar(value="0.0 FPS")
        self.total_var = tk.StringVar(value="0")
        self.with_mask_var = tk.StringVar(value="0")
        self.without_mask_var = tk.StringVar(value="0")
        self.incorrect_var = tk.StringVar(value="0")
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH))

        self._build_ui()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(self.root, text="Điều khiển")
        controls.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))
        controls.columnconfigure(7, weight=1)

        ttk.Button(controls, text="Chọn ảnh", command=self.choose_image).grid(
            row=0, column=0, padx=(8, 4), pady=8
        )
        ttk.Button(controls, text="Mở video", command=self.open_video).grid(
            row=0, column=1, padx=4, pady=8
        )
        ttk.Button(controls, text="Webcam", command=self.start_webcam).grid(
            row=0, column=2, padx=4, pady=8
        )
        ttk.Button(controls, text="Dừng", command=self.stop_stream).grid(
            row=0, column=3, padx=4, pady=8
        )
        ttk.Button(controls, text="Lưu ảnh kết quả", command=self.save_result).grid(
            row=0, column=4, padx=4, pady=8
        )

        ttk.Label(controls, text="Confidence:").grid(
            row=0, column=5, padx=(18, 4), pady=8
        )
        confidence_scale = ttk.Scale(
            controls,
            from_=0.10,
            to=0.90,
            variable=self.confidence_var,
            command=self._on_confidence_changed,
            length=150,
        )
        confidence_scale.grid(row=0, column=6, padx=4, pady=8)
        ttk.Label(controls, textvariable=self.confidence_text, width=5).grid(
            row=0, column=7, sticky="w", padx=(0, 8), pady=8
        )
        ttk.Button(controls, text="Phân tích lại", command=self.reanalyze_image).grid(
            row=0, column=8, padx=(4, 8), pady=8
        )

        content = ttk.Frame(self.root)
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(content, text="Kết quả phát hiện")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview = tk.Label(
            preview_frame,
            text="Chọn ảnh, video hoặc bật webcam để bắt đầu",
            bg="#111827",
            fg="#d1d5db",
            font=("Segoe UI", 14),
            justify="center",
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        info_frame = ttk.LabelFrame(content, text="Thông tin")
        info_frame.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(
            info_frame,
            text="MÔ HÌNH",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 2))
        ttk.Label(
            info_frame,
            textvariable=self.model_var,
            wraplength=260,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 12))

        ttk.Separator(info_frame).grid(row=2, column=0, sticky="ew", padx=12)
        ttk.Label(
            info_frame,
            text="NGUỒN",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=3, column=0, sticky="w", padx=12, pady=(12, 2))
        ttk.Label(
            info_frame,
            textvariable=self.source_var,
            wraplength=260,
            justify="left",
        ).grid(row=4, column=0, sticky="w", padx=12, pady=(0, 12))

        ttk.Label(
            info_frame,
            text="THỐNG KÊ KHUÔN MẶT",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=5, column=0, sticky="w", padx=12, pady=(4, 6))
        stats = ttk.Frame(info_frame)
        stats.grid(row=6, column=0, sticky="ew", padx=12)
        stats.columnconfigure(1, weight=1)
        self._add_stat_row(stats, 0, "Tổng số", self.total_var)
        self._add_stat_row(stats, 1, "Có khẩu trang", self.with_mask_var)
        self._add_stat_row(stats, 2, "Không khẩu trang", self.without_mask_var)
        self._add_stat_row(stats, 3, "Đeo sai khẩu trang", self.incorrect_var)
        self._add_stat_row(stats, 4, "Tốc độ", self.fps_var)

        ttk.Separator(info_frame).grid(row=7, column=0, sticky="ew", padx=12, pady=12)
        ttk.Label(
            info_frame,
            text=(
                "Ứng dụng hỗ trợ:\n"
                "• Object Detection: vẽ vùng khuôn mặt\n"
                "• Facial Analysis: phân loại trạng thái khẩu trang\n"
                "• Real-time Detection: xử lý webcam/video"
            ),
            justify="left",
            wraplength=270,
        ).grid(row=8, column=0, sticky="w", padx=12, pady=(0, 12))

        status = ttk.Label(
            self.root,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
        )
        status.grid(row=2, column=0, sticky="ew", padx=12, pady=(6, 10))

    @staticmethod
    def _add_stat_row(
        parent: ttk.Frame,
        row: int,
        label: str,
        value: tk.StringVar,
    ) -> None:
        ttk.Label(parent, text=f"{label}:").grid(
            row=row, column=0, sticky="w", pady=3
        )
        ttk.Label(parent, textvariable=value, font=("Segoe UI", 10, "bold")).grid(
            row=row, column=1, sticky="e", pady=3
        )

    def _load_model(self) -> None:
        try:
            self.detector = MaskDetector(DEFAULT_MODEL_PATH)
            class_list = ", ".join(self.detector.class_names.values())
            self.status_var.set(
                f"Đã load model | input {self.detector.input_width}x{self.detector.input_height} "
                f"| classes: {class_list}"
            )
        except Exception as error:  # noqa: BLE001 - show a useful GUI error.
            self.status_var.set("Không thể load model")
            messagebox.showerror("Lỗi model", str(error), parent=self.root)

    def _on_confidence_changed(self, value: str) -> None:
        self.confidence_text.set(f"{float(value):.2f}")

    def _set_source(self, source: str) -> None:
        self.source_kind = source
        self.source_var.set(source)

    def _reset_stats(self) -> None:
        self.total_var.set("0")
        self.with_mask_var.set("0")
        self.without_mask_var.set("0")
        self.incorrect_var.set("0")
        self.fps_var.set("0.0 FPS")

    def _update_stats(self, counts: Counter[str]) -> None:
        self.total_var.set(str(sum(counts.values())))
        self.with_mask_var.set(str(counts.get("with_mask", 0)))
        self.without_mask_var.set(str(counts.get("without_mask", 0)))
        self.incorrect_var.set(str(counts.get("mask_weared_incorrect", 0)))
        self.fps_var.set(f"{self.fps:.1f} FPS")

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 680), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def _process_frame(self, frame: np.ndarray, is_stream: bool = False) -> None:
        if self.detector is None:
            return

        detections = self.detector.detect(
            frame,
            confidence_threshold=float(self.confidence_var.get()),
        )
        annotated, counts = draw_detections(frame, detections)
        self.last_input_frame = frame.copy()
        self.last_annotated_frame = annotated

        if is_stream:
            now = time.perf_counter()
            elapsed = now - self.last_frame_time
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                self.fps = instantaneous_fps if self.fps == 0 else self.fps * 0.85 + instantaneous_fps * 0.15
            self.last_frame_time = now
        else:
            self.fps = 0.0

        self._update_stats(counts)
        self._show_frame(annotated)
        self.status_var.set(
            f"{self.source_kind} | Đã phát hiện {len(detections)} khuôn mặt | "
            f"confidence >= {float(self.confidence_var.get()):.2f}"
        )

    def choose_image(self) -> None:
        self.stop_stream()
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh cần phát hiện",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        frame = cv2.imread(file_path)
        if frame is None:
            messagebox.showerror("Lỗi ảnh", "Không thể đọc file ảnh này.", parent=self.root)
            return

        self.source_path = Path(file_path)
        self._set_source(f"Ảnh: {self.source_path.name}")
        try:
            self._process_frame(frame)
        except Exception as error:  # noqa: BLE001 - present inference errors in GUI.
            messagebox.showerror("Lỗi inference", str(error), parent=self.root)

    def reanalyze_image(self) -> None:
        if self.last_input_frame is None or self.stream_running:
            self.status_var.set("Hãy chọn một ảnh trước khi dùng Phân tích lại.")
            return
        try:
            self._process_frame(self.last_input_frame.copy())
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("Lỗi inference", str(error), parent=self.root)

    def open_video(self) -> None:
        self.stop_stream()
        file_path = filedialog.askopenfilename(
            title="Chọn video cần phát hiện",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        capture = cv2.VideoCapture(file_path)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Lỗi video", "Không thể mở video này.", parent=self.root)
            return

        self._start_stream(capture, f"Video: {Path(file_path).name}", "video")

    def start_webcam(self) -> None:
        self.stop_stream()
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror(
                "Lỗi webcam",
                "Không thể mở webcam mặc định (camera index 0).",
                parent=self.root,
            )
            return

        self._start_stream(capture, "Webcam (camera 0)", "webcam")

    def _start_stream(
        self,
        capture: cv2.VideoCapture,
        source: str,
        source_kind: str,
    ) -> None:
        self.capture = capture
        self.stream_running = True
        self.source_path = None
        self._set_source(source)
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0
        self._reset_stats()
        self._read_stream_frame()

    def _read_stream_frame(self) -> None:
        if not self.stream_running or self.capture is None:
            return

        success, frame = self.capture.read()
        if not success:
            self.stop_stream()
            self.status_var.set("Đã kết thúc video hoặc không đọc được frame tiếp theo.")
            return

        try:
            self._process_frame(frame, is_stream=True)
        except Exception as error:  # noqa: BLE001
            self.stop_stream()
            messagebox.showerror("Lỗi inference", str(error), parent=self.root)
            return

        self.after_id = self.root.after(10, self._read_stream_frame)

    def stop_stream(self) -> None:
        self.stream_running = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
            self.after_id = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def save_result(self) -> None:
        if self.last_annotated_frame is None:
            messagebox.showinfo(
                "Chưa có kết quả",
                "Hãy phân tích một ảnh, video hoặc webcam trước.",
                parent=self.root,
            )
            return

        file_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png")],
        )
        if not file_path:
            return

        if cv2.imwrite(file_path, self.last_annotated_frame):
            self.status_var.set(f"Đã lưu kết quả: {file_path}")
        else:
            messagebox.showerror("Lỗi lưu file", "Không thể lưu ảnh kết quả.", parent=self.root)

    def close(self) -> None:
        self.stop_stream()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = MaskDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
