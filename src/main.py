import argparse
import time
from pathlib import Path
import cv2
from src.tracker import TrafficTracker
from src.visualizer import TrafficVisualizer


def parse_args() -> argparse.Namespace:
    """Define y lee los argumentos pasados por terminal al ejecutar el script."""
    parser = argparse.ArgumentParser(
        description="Real-time Traffic Flow Analytics & GDPR Anonymization with YOLO11 + ByteTrack"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="assets/traffic.mp4",
        help="Ruta al vídeo de entrada, URL de stream o índice de webcam (ej. '0').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/traffic_analyzed.mp4",
        help="Ruta donde se guardará el vídeo procesado.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Pesos del modelo YOLO11 (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Umbral mínimo de confianza para las detecciones (0.0 a 1.0).",
    )
    parser.add_argument(
        "--line-pos",
        type=float,
        default=0.60,
        help="Posición vertical relativa de la línea de conteo (0.0 arriba, 1.0 abajo).",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Activa el difuminado RGPD en tiempo real para peatones y matrículas.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Procesa el vídeo sin abrir ventana en pantalla (útil para renderizar más rápido).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Si el usuario pasa "0" como string, lo convertimos a entero para abrir la webcam
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de vídeo: {args.source}")

    # Obtenemos propiedades del vídeo original para configurar el archivo de salida
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Calculamos la coordenada Y exacta en píxeles de la línea virtual
    line_y = int(height * args.line_pos)

    # Creamos la carpeta de salida si no existe y preparamos el VideoWriter
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_video, (width, height))

    # Instanciamos nuestros dos módulos principales
    tracker = TrafficTracker(model_name=args.model, conf_threshold=args.conf)
    visualizer = TrafficVisualizer(anonymize=args.anonymize)

    print(f"[INFO] Procesando '{args.source}' ({width}x{height} @ {fps_video:.1f} FPS)...")
    print(f"[INFO] Modelo: {args.model} | Anonimización RGPD: {'ON' if args.anonymize else 'OFF'}")
    print("[INFO] Pulsa 'Q' en la ventana de vídeo para detener antes de tiempo.")

    prev_time = time.perf_counter()

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Análisis: detección + tracking + conteo
            tracked_objects, stats = tracker.process_frame(frame, line_y=line_y)

            # Cálculo de FPS reales de procesamiento
            curr_time = time.perf_counter()
            fps_real = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # Renderizado: privacidad + estelas + cajas + HUD
            annotated_frame = visualizer.draw(
                frame=frame,
                tracked_objects=tracked_objects,
                track_history=tracker.track_history,
                stats=stats,
                line_y=line_y,
                fps=fps_real,
            )

            # Guardamos el fotograma en el archivo de salida
            writer.write(annotated_frame)

            # Mostramos la ventana en directo salvo que se haya pasado --no-show
            if not args.no_show:
                cv2.imshow("YOLO11 Traffic Analytics", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Interrumpido por el usuario.")
                    break
    finally:
        # Liberamos siempre los recursos de hardware y archivos aunque ocurra un error
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"[OK] Vídeo guardado en: {output_path}")
    print(f"[RESUMEN] Down: {tracker.stats.total_down} | Up: {tracker.stats.total_up} | Desglose: {dict(tracker.stats.counts_by_class)}")


if __name__ == "__main__":
    main()