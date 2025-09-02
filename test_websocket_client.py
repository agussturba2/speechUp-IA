#!/usr/bin/env python3
"""
Cliente de prueba para el WebSocket de análisis de oratoria.
Este script captura video de la webcam y lo envía al servidor WebSocket.
"""

import asyncio
import cv2
import websockets
import json
import time
import numpy as np
import argparse
from typing import Optional

async def test_oratory_websocket(
    server_url: str = "ws://localhost:8000/ws/v1/feedback-oratoria",
    camera_id: int = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    test_duration: Optional[int] = None
):
    """
    Prueba el WebSocket de análisis de oratoria enviando frames de video.
    
    Args:
        server_url: URL del servidor WebSocket
        camera_id: ID de la cámara a utilizar
        width: Ancho de los frames
        height: Alto de los frames
        fps: Frames por segundo a enviar
        test_duration: Duración de la prueba en segundos (None para continuar hasta Ctrl+C)
    """
    # Inicializar la cámara
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {camera_id}")
        return
    
    print(f"Conectando a {server_url}...")
    
    try:
        async with websockets.connect(server_url) as websocket:
            print("Conexión establecida. Enviando frames de video...")
            
            start_time = time.time()
            frame_count = 0
            frame_interval = 1.0 / fps
            
            # Bucle principal
            while True:
                # Verificar si se ha alcanzado la duración de la prueba
                if test_duration and (time.time() - start_time) > test_duration:
                    print(f"Prueba completada después de {test_duration} segundos")
                    break
                
                # Capturar frame
                ret, frame = cap.read()
                if not ret:
                    print("Error al capturar frame")
                    break
                
                # Redimensionar frame si es necesario
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                # Codificar frame como JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                
                # Enviar frame al servidor
                await websocket.send(frame_data)
                frame_count += 1
                
                # Mostrar frame localmente
                cv2.imshow('Enviando al WebSocket', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Recibir respuesta del servidor (no bloqueante)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    if isinstance(response, str):
                        # Respuesta JSON
                        data = json.loads(response)
                        print(f"Recibido: {data.get('status', 'unknown')}")
                        
                        # Si se recibió feedback completo, mostrarlo
                        if data.get('status') == 'complete' and 'feedback' in data:
                            feedback = data['feedback']
                            scores = feedback.get('scores', {})
                            print("\n=== RESULTADOS DE ANÁLISIS ===")
                            print(f"Engagement: {scores.get('engagement', 'N/A')}")
                            print(f"Confianza: {scores.get('delivery_confidence', 'N/A')}")
                            print(f"Fluidez: {scores.get('fluency', 'N/A')}")
                            print("============================\n")
                except asyncio.TimeoutError:
                    # No hay respuesta disponible, continuar
                    pass
                
                # Mantener la tasa de frames
                await asyncio.sleep(frame_interval)
                
                # Mostrar FPS cada segundo
                if frame_count % fps == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    print(f"FPS: {current_fps:.2f}")
    
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Conexión cerrada: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Liberar recursos
        cap.release()
        cv2.destroyAllWindows()
        print("Cliente finalizado")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente de prueba para WebSocket de análisis de oratoria")
    parser.add_argument("--url", default="ws://localhost:8000/ws/v1/feedback-oratoria", 
                        help="URL del servidor WebSocket")
    parser.add_argument("--camera", type=int, default=0, 
                        help="ID de la cámara a utilizar")
    parser.add_argument("--width", type=int, default=640, 
                        help="Ancho de los frames")
    parser.add_argument("--height", type=int, default=480, 
                        help="Alto de los frames")
    parser.add_argument("--fps", type=int, default=30, 
                        help="Frames por segundo a enviar")
    parser.add_argument("--duration", type=int, default=None, 
                        help="Duración de la prueba en segundos")
    
    args = parser.parse_args()
    
    asyncio.run(test_oratory_websocket(
        server_url=args.url,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        test_duration=args.duration
    ))
