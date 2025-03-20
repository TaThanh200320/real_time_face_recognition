import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

def create_face_database_from_directory(db_directory, output_path):
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    embeddings = []
    valid_names = []
    
    for person_dir in os.listdir(db_directory):
        person_path = os.path.join(db_directory, person_dir)
        
        if not os.path.isdir(person_path):
            continue
        
        person_name = person_dir
        image_count = 0
        for image_file in os.listdir(person_path):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(person_path, image_file)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image from: {img_path}")
                    continue
                    
                faces = app.get(img)
                if not faces:
                    print(f"Cannot detect faces from: {img_path}")
                    continue
                
                face_embedding = faces[0].embedding
                embeddings.append(face_embedding)
                valid_names.append(person_name)
                image_count += 1
                print(f"Added face of {person_name} from photo {image_file}")
                
            except Exception as e:
                print(f"Processing error {img_path}: {e}")
        
        print(f"Processed {image_count} images for {person_name}")
    
    database = {
        'embeddings': embeddings,
        'names': valid_names
    }
    np.save(output_path, database)
    print(f"Saved {len(valid_names)} face into {output_path}")
    return len(valid_names)

if __name__ == '__main__':
    db_directory = "db"
    num_faces = create_face_database_from_directory(db_directory, "face_database.npy")
    print(f"Done! Added a total of {num_faces} faces to the database.")