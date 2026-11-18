from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from ultralytics import YOLO

# Diccionario con los IDs oficiales del dataset COCO (preentrenado en YOLO).
# Filtramos únicamente las clases que nos interesan para analizar tráfico
# e incluimos 'Person' (0) para poder difuminar peatones por privacidad (RGPD).
TRAFFIC_CLASSES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


@dataclass
class TrackedObject:
    """Representa un único objeto detectado y seguido en el fotograma actual."""
    track_id: int                    # Identificador numérico único asignado por el tracker (ej. coche #4)
    class_id: int                    # ID numérico de la clase en COCO (ej. 2 para coche)
    class_name: str                  # Nombre legible de la clase (ej. "Car")
    bbox: Tuple[int, int, int, int]  # Coordenadas de la caja: (x1, y1) arriba-izq, (x2, y2) abajo-der
    center: Tuple[int, int]          # Punto central (centroide) de la caja: (cx, cy)
    confidence: float                # Nivel de seguridad de la detección (entre 0.0 y 1.0)


@dataclass
class TrafficStats:
    """Almacena y acumula las métricas globales de tráfico a lo largo de todo el vídeo."""
    total_down: int = 0              # Total de vehículos que han cruzado la línea hacia abajo
    total_up: int = 0                # Total de vehículos que han cruzado la línea hacia arriba
    # Diccionario que cuenta cuántos vehículos de cada tipo han cruzado (ej. {"Car": 12, "Bus": 1}).
    # field(default_factory=...) asegura que cada nueva instancia tenga su propio diccionario vacío.
    counts_by_class: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    active_vehicles: int = 0         # Número de vehículos visibles en pantalla en este instante


class TrafficTracker:
    def __init__(self, model_name: str = "yolo11m.pt", conf_threshold: float = 0.35, trail_length: int = 30):
        # Carga los pesos de YOLO11. Si el archivo .pt no existe en local, lo descarga solo.
        # Ultralytics detecta automáticamente si hay una GPU NVIDIA (CUDA) y la usa.
        self.model = YOLO(model_name)
        
        # Umbral mínimo de confianza (0.35 = 35%). Todo lo que esté por debajo se descarta.
        self.conf_threshold = conf_threshold
        
        # Número máximo de puntos pasados que guardaremos por vehículo para dibujar su estela.
        self.trail_length = trail_length

        # Memoria espacial: mapea cada 'track_id' -> lista con sus últimas coordenadas (cx, cy).
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        
        # Conjunto (set) con los IDs que ya han cruzado la línea virtual.
        # Evita que un coche parado encima de la línea se cuente varias veces seguidas.
        self.counted_ids: set[int] = set()
        
        # Inicializa el objeto que llevará el recuento total.
        self.stats = TrafficStats()

    def process_frame(self, frame: np.ndarray, line_y: int) -> Tuple[List[TrackedObject], TrafficStats]:
        """Ejecuta detección + seguimiento en un fotograma y actualiza los contadores de cruce."""
        
        # Ejecutamos la inferencia con seguimiento integrado:
        # - persist=True: mantiene el estado de los objetos entre fotogramas para conservar sus IDs.
        # - tracker="bytetrack.yaml": usa el algoritmo ByteTrack (muy robusto ante oclusiones).
        # - classes=...: filtra en GPU solo las clases definidas en TRAFFIC_CLASSES.
        # - verbose=False: evita que YOLO imprima estadísticas por consola en cada fotograma.
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(TRAFFIC_CLASSES.keys()),
            conf=self.conf_threshold,
            imgsz=1280,     # Aumenta la resolución de visión del modelo (por defecto es 640).
            verbose=False,
        )

        tracked_objects: List[TrackedObject] = []
        active_vehicles_count = 0

        # Extraemos las cajas detectadas del primer (y único) fotograma enviado.
        boxes = results[0].boxes
        
        # Comprobamos que haya detecciones y que el tracker ya haya asignado IDs válidos
        # (en el primer fotograma del vídeo, boxes.id puede ser None).
        if boxes is not None and boxes.id is not None:
            # Movemos los tensores de la VRAM de la GPU a la RAM de la CPU (.cpu())
            # y los convertimos a listas estándar de Python (.tolist()) para iterar rápido.
            xyxy_list = boxes.xyxy.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.float().cpu().tolist()

            # Recorremos cada objeto detectado empaquetando sus 4 atributos paralelos
            for bbox, track_id, cls_id, conf in zip(xyxy_list, track_ids, class_ids, confs):
                x1, y1, x2, y2 = bbox
                
                # Calculamos el centroide (punto medio) de la caja con división entera
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cls_name = TRAFFIC_CLASSES.get(cls_id, "Unknown")

                # Creamos el objeto limpio desacoplado de PyTorch/Ultralytics
                obj = TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    confidence=conf,
                )
                tracked_objects.append(obj)

                # Guardamos estelas y contamos cruces SOLO para vehículos (excluimos Person, ID 0)
                if cls_id != 0:
                    active_vehicles_count += 1
                    
                    # Añadimos el centroide actual al historial de este vehículo
                    history = self.track_history[track_id]
                    history.append((cx, cy))
                    
                    # Si la estela supera el límite (ej. 30 puntos), borramos el punto más antiguo
                    if len(history) > self.trail_length:
                        history.pop(0)

                    # Necesitamos al menos 2 puntos (fotograma anterior y actual) para detectar movimiento,
                    # y comprobamos que este ID no haya sido contado ya anteriormente.
                    if len(history) >= 2 and track_id not in self.counted_ids:
                        prev_y = history[-2][1]  # Coordenada Y en el fotograma anterior
                        curr_y = history[-1][1]  # Coordenada Y en el fotograma actual

                        # En OpenCV, Y=0 está arriba y crece hacia abajo.
                        # Cruce hacia abajo: estaba por encima de la línea y ahora está en o por debajo.
                        if prev_y < line_y <= curr_y:
                            self.stats.total_down += 1
                            self.stats.counts_by_class[cls_name] += 1
                            self.counted_ids.add(track_id)
                            
                        # Cruce hacia arriba: estaba por debajo de la línea y ahora está en o por encima.
                        elif prev_y > line_y >= curr_y:
                            self.stats.total_up += 1
                            self.stats.counts_by_class[cls_name] += 1
                            self.counted_ids.add(track_id)

        # Actualizamos el contador de vehículos presentes en el fotograma actual
        self.stats.active_vehicles = active_vehicles_count
        return tracked_objects, self.stats