import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis
import threading
from collections import deque

class RTSPFaceRecognition:
    def __init__(self, rtsp_url, threshold=0.5, face_db_path=None):
        self.rtsp_url = rtsp_url
        self.threshold = threshold
        self.face_db_path = face_db_path
        
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.known_faces = []
        self.known_names = []
        if self.face_db_path:
            self._load_face_database()
        
        self.running = False
        self.frame_queue = deque(maxlen=5)
        self.frame_count = 0
        self.fps_start_time = time.monotonic()
        self.fps = 0
        
    def _load_face_database(self):
        try:
            database = np.load(self.face_db_path, allow_pickle=True).item()
            self.known_faces = database.get('embeddings', [])
            self.known_names = database.get('names', [])
            print(f"Uploaded {len(self.known_names)} faces from database")
        except Exception as e:
            print(f"Error when uploading faces: {e}")
    
    def _calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def _recognize_face(self, face_embedding):
        if not self.known_faces:
            return "Unknown"
        
        similarities = [self._calculate_similarity(face_embedding, known_face) for known_face in self.known_faces]
        max_similarity_idx = np.argmax(similarities)
        
        if similarities[max_similarity_idx] >= self.threshold:
            return self.known_names[max_similarity_idx]
        return "Unknown"
    
    def _process_frame(self, frame):
        if frame is None:
            return frame
        
        faces = self.app.get(frame)
        self.frame_count += 1
        current_time = time.monotonic()
        elapsed_time = current_time - self.fps_start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.fps_start_time = current_time
        
        for face in faces:
            bbox = face.bbox.astype(np.int32)
            face_embedding = face.embedding
            name = self._recognize_face(face_embedding)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            confidence = face.det_score
            label = f"{name} ({confidence:.2f})"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
    
    def _capture_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            print(f"Cannot open rtsp: {self.rtsp_url}")
            return
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Error when reading rtsp frame")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(self.rtsp_url)
                continue
            
            self.frame_queue.append(frame.copy())
        
        cap.release()
    
    def _display_frames(self):
        while self.running:
            if not self.frame_queue:
                time.sleep(0.01)
                continue
            
            frame = self.frame_queue.popleft()
            processed_frame = self._process_frame(frame)
            
            cv2.imshow("Face Recognition", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
    
    def start(self):
        self.running = True
        capture_thread = threading.Thread(target=self._capture_frames)
        display_thread = threading.Thread(target=self._display_frames)
        capture_thread.start()
        display_thread.start()
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.running = False
        
        capture_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":  
    rtsp_url = "rtsp://admin:Stc@vielina.com@192.168.8.193:554/Streaming/channels/101"
    
    face_recognition = RTSPFaceRecognition(
        rtsp_url=rtsp_url,
        threshold=0.5,
        face_db_path="face_database.npy"
    )
    
    face_recognition.start()