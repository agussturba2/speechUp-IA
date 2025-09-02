"""
Test client for real-time video analysis WebSocket API
"""

import asyncio
import websockets
import cv2
import json
import time
import argparse

async def send_video_frames(video_path, websocket_uri='ws://localhost:8000/feedback-realtime'):
    """
    Sends video frames to the WebSocket server and displays feedback in real-time
    
    Args:
        video_path: Path to video file to stream
        websocket_uri: URI of the WebSocket endpoint
    """
    print(f"Connecting to {websocket_uri}...")
    
    try:
        async with websockets.connect(websocket_uri) as websocket:
            print("Connected! Streaming video frames...")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_path}")
                return
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25
                
            # Calculate delay between frame sends to simulate real-time
            frame_delay = 1.0 / fps
            
            # Loop through video frames
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame to JPEG format for sending
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                
                # Send frame through WebSocket
                await websocket.send(frame_data)
                frame_count += 1
                
                # Handle incoming messages (non-blocking)
                try:
                    # Use very short timeout to make this non-blocking
                    response = await asyncio.wait_for(websocket.recv(), 0.01)
                    feedback = json.loads(response)
                    
                    # Display feedback
                    print("\n--- Real-time Feedback ---")
                    print(f"Status: {feedback.get('estado', 'N/A')}")
                    print(f"Faces detected: {feedback.get('rostros_detectados', 0)}/{feedback.get('frames_procesados', 0)} frames")
                    print(f"Good posture: {feedback.get('porcentaje_posturas_buenas', 0)}%")
                    print(f"Feedback: {feedback.get('feedback_general', 'Sin feedback')}")
                    print("-------------------------\n")
                    
                except asyncio.TimeoutError:
                    # No message available, continue sending frames
                    pass
                
                # Simulate real-time by adding delay between frames
                await asyncio.sleep(frame_delay)
                
                # Display progress
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"Sent {frame_count} frames in {elapsed:.1f} seconds")
            
            # Wait for final feedback
            try:
                final_response = await asyncio.wait_for(websocket.recv(), 2.0)
                final_feedback = json.loads(final_response)
                print("\n=== Final Feedback ===")
                print(f"Status: {final_feedback.get('estado', 'N/A')}")
                print(f"Overall: {final_feedback.get('feedback_general', 'Sin feedback')}")
                print("=====================\n")
            except asyncio.TimeoutError:
                print("No final feedback received")
            
            print(f"Finished streaming {frame_count} frames")
            cap.release()
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test client for real-time video analysis")
    parser.add_argument("video_path", help="Path to video file to analyze")
    parser.add_argument("--uri", default="ws://localhost:8000/feedback-realtime", 
                        help="WebSocket URI (default: ws://localhost:8000/feedback-realtime)")
    
    args = parser.parse_args()
    
    asyncio.run(send_video_frames(args.video_path, args.uri))
