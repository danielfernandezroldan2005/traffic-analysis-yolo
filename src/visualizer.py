from typing import Dict, List, Tuple
import cv2
import numpy as np
from src.tracker import TrackedObject, TrafficStats

# Paleta de colores en formato BGR (Azul, Verde, Rojo) de OpenCV.
# Elegimos tonos vibrantes que contrasten bien sobre el asfalto gris.
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Car": (255, 191, 0),         # Azul eléctrico / cian intenso
    "Motorcycle": (0, 215, 255),  # Amarillo dorado
    "Bus": (180, 105, 255),       # Rosa / magenta neón
    "Truck": (50, 205, 50),       # Verde lima brillante
    "Person": (200, 200, 200),    # Gris claro (solo si no se difumina)
}


class TrafficVisualizer:
    """Se encarga de renderizar toda la capa gráfica (estelas, cajas, HUD y privacidad) sobre el vídeo."""

    def __init__(self, anonymize: bool = False):
        # Si es True, difumina peatones y la zona de matrículas de vehículos (cumplimiento RGPD)
        self.anonymize = anonymize

    def _apply_privacy_blur(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> None:
        """Difumina peatones y zonas de matrículas directamente sobre el frame (in-place)."""
        h_img, w_img, _ = frame.shape

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            # Aseguramos que las coordenadas no se salgan de los bordes de la imagen
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            if obj.class_id == 0:
                # Peatón: difuminamos toda la caja delimitadora
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (35, 35), 30)
            else:
                # Vehículo: difuminamos solo el tercio inferior (zona habitual de matrículas)
                box_height = y2 - y1
                plate_y1 = y1 + int(box_height * 0.65)
                if y2 > plate_y1:
                    roi = frame[plate_y1:y2, x1:x2]
                    frame[plate_y1:y2, x1:x2] = cv2.GaussianBlur(roi, (25, 25), 20)

    def _draw_trails_and_boxes(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        track_history: Dict[int, List[Tuple[int, int]]],
    ) -> None:
        """Dibuja las estelas de movimiento degradadas y las cajas con etiqueta tipo píldora."""
        for obj in tracked_objects:
            # Si el modo privacidad está activo, no dibujamos caja sobre peatones ya difuminados
            if self.anonymize and obj.class_id == 0:
                continue

            color = CLASS_COLORS.get(obj.class_name, (255, 255, 255))
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.center

            # 1. Dibujar la estela de trayectoria (solo para vehículos con historial)
            history = track_history.get(obj.track_id, [])
            if len(history) >= 2:
                for i in range(1, len(history)):
                    # El grosor aumenta progresivamente desde la cola (1px) hasta el vehículo (3px)
                    thickness = max(1, int((i / len(history)) * 3))
                    cv2.line(frame, history[i - 1], history[i], color, thickness, lineType=cv2.LINE_AA)

            # 2. Dibujar la caja delimitadora y el punto central (centroide)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, color, -1, lineType=cv2.LINE_AA)

            # 3. Etiqueta superior limpia: "#ID Clase Confianza" (ej. "#12 Car 0.91")
            label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Fondo relleno para que el texto siempre se lea bien sobre cualquier fondo
            label_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(frame, (x1, label_y1), (x1 + text_w + 8, y1), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 15, 15),  # Texto oscuro sobre fondo brillante para máximo contraste
                1,
                lineType=cv2.LINE_AA,
            )

    def _draw_hud(self, frame: np.ndarray, stats: TrafficStats, line_y: int, fps: float) -> None:
        """Dibuja la línea virtual de conteo y el panel HUD semitransparente con las métricas."""
        h_img, w_img, _ = frame.shape

        # Línea virtual horizontal de conteo
        cv2.line(frame, (0, line_y), (w_img, line_y), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            "COUNTING LINE",
            (15, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )

        # Caja semitransparente arriba a la izquierda (efecto cristal oscuro)
        overlay = frame.copy()
        panel_w, panel_h = 270, 175
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 25), -1)
        # Mezclamos el rectángulo negro al 70% de opacidad con el fotograma original
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (80, 80, 90), 1)

        # Métricas dentro del HUD
        lines = [
            f"FPS: {fps:.1f}  |  Active: {stats.active_vehicles}",
            f"Down: {stats.total_down}   |  Up: {stats.total_up}",
            "-" * 26,
            f"Cars: {stats.counts_by_class['Car']}   Motorcycles: {stats.counts_by_class['Motorcycle']}",
            f"Buses: {stats.counts_by_class['Bus']}  Trucks: {stats.counts_by_class['Truck']}",
        ]

        y_offset = 42
        for idx, text in enumerate(lines):
            # Destacamos las dos primeras líneas en blanco/cian y el resto en gris claro
            text_color = (0, 255, 255) if idx == 1 else (240, 240, 240)
            cv2.putText(
                frame,
                text,
                (28, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                text_color,
                1,
                lineType=cv2.LINE_AA,
            )
            y_offset += 28

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        track_history: Dict[int, List[Tuple[int, int]]],
        stats: TrafficStats,
        line_y: int,
        fps: float,
    ) -> np.ndarray:
        """Orquesta todo el renderizado sobre una copia limpia del fotograma."""
        annotated = frame.copy()

        # Si está activado, difuminamos matrículas/peatones ANTES de pintar encima
        if self.anonymize:
            self._apply_privacy_blur(annotated, tracked_objects)

        # Estelas y cajas
        self._draw_trails_and_boxes(annotated, tracked_objects, track_history)

        # Línea de conteo y panel HUD
        self._draw_hud(annotated, stats, line_y, fps)

        return annotated