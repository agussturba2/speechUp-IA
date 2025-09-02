#!/usr/bin/env python3
"""
Cliente para enviar un video completo al WebSocket de análisis de oratoria.
"""

import asyncio
import websockets
import json
import argparse
import os
from typing import Optional

async def send_video_to_websocket(
    video_path: str,
    server_url: str = "ws://localhost:8000/ws/v1/feedback-oratoria"
):
    """
    Envía un video completo al WebSocket y recibe actualizaciones.
    
    Args:
        video_path: Ruta al archivo de video
        server_url: URL del servidor WebSocket
    """
    if not os.path.exists(video_path):
        print(f"Error: El archivo {video_path} no existe")
        return
    
    # Leer el archivo de video
    with open(video_path, "rb") as f:
        video_data = f.read()
    
    # Agregar prefijo VID
    video_data_with_prefix = b'VID' + video_data
    
    print(f"Conectando a {server_url}...")
    
    try:
        async with websockets.connect(server_url) as websocket:
            print(f"Conectado a {server_url}")
            
            # Recibir mensaje inicial de conexión
            try:
                response = await websocket.recv()
                print(f"Mensaje inicial: {response}")
            except Exception as e:
                print(f"No se recibió mensaje inicial: {e}")
            
            # Enviar video completo
            print(f"Enviando video de {len(video_data_with_prefix)} bytes...")
            await websocket.send(video_data_with_prefix)
            
            # Recibir actualizaciones
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    status = data.get("status", "unknown")
                    
                    print(f"\n--- Actualización: {status} ---")
                    
                    if status == "info":
                        print(f"Total frames: {data.get('total_frames')}")
                        print(f"FPS: {data.get('fps')}")
                    
                    elif status == "processing":
                        print(f"Mensaje: {data.get('message')}")
                    
                    elif status == "progress":
                        print(f"Progreso: {data.get('progress_percent')}%")
                        print(f"Chunk: {data.get('chunk')}/{data.get('total_chunks')}")
                        
                        # Mostrar algunos resultados parciales si están disponibles
                        if "feedback" in data and "scores" in data["feedback"]:
                            scores = data["feedback"]["scores"]
                            print("\nResultados parciales:")
                            for key, value in scores.items():
                                print(f"  - {key}: {value}")
                    
                    elif status == "complete":
                        print("¡Análisis completo!")
                        
                        # Mostrar resultados finales
                        if "feedback" in data and "scores" in data["feedback"]:
                            scores = data["feedback"]["scores"]
                            print("\nResultados finales:")
                            for key, value in scores.items():
                                print(f"  - {key}: {value}")
                        
                        # Guardar resultados completos en un archivo JSON
                        with open("resultados_analisis.json", "w") as f:
                            json.dump(data["feedback"], f, indent=2)
                        print("\nResultados guardados en 'resultados_analisis.json'")
                        
                        break
                    
                    elif status == "error":
                        print(f"Error: {data.get('message')}")
                        break
                    
                except websockets.exceptions.ConnectionClosed:
                    print("Conexión cerrada")
                    break
                except json.JSONDecodeError:
                    print(f"Respuesta no JSON: {response}")
                except Exception as e:
                    print(f"Error: {e}")
                    break
    
    except Exception as e:
        print(f"Error de conexión: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente para enviar video al WebSocket")
    parser.add_argument("video_path", help="Ruta al archivo de video")
    parser.add_argument("--url", default="ws://localhost:8000/ws/v1/feedback-oratoria", 
                        help="URL del servidor WebSocket")
    
    args = parser.parse_args()
    
    asyncio.run(send_video_to_websocket(
        video_path=args.video_path,
        server_url=args.url
    ))
