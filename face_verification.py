import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import sys
import json
import base64
import os
import traceback

# Enable unsafe deserialization to allow Lambda layers
tf.keras.config.enable_unsafe_deserialization()

# Define the custom layer that was used during training
class L2Normalize(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super(L2Normalize, self).get_config()

class SafeFaceVerifier:
    def __init__(self, model_path='face_verification_model.h5'):
        print("🚀 Initializing Safe Face Verifier...", file=sys.stderr)
        self.model = None
        self.input_shape = (128, 128, 3)
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the TensorFlow model with safe_mode=False"""
        try:
            if not os.path.exists(model_path):
                print(f"❌ Model file not found: {model_path}", file=sys.stderr)
                return
            
            print(f"📁 Loading model from: {model_path}", file=sys.stderr)
            
            # Register custom layers
            custom_objects = {
                'L2Normalize': L2Normalize
            }
            
            # Load with safe_mode=False to allow Lambda layers
            self.model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False  # This allows Lambda layers
            )
            print("✅ Model loaded successfully with unsafe deserialization!", file=sys.stderr)
            print(f"📐 Model input shape: {self.model.input_shape}", file=sys.stderr)
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}", file=sys.stderr)
            self.model = None
    
    def decode_base64_image(self, image_data):
        """Decode base64 image"""
        try:
            if not image_data or not isinstance(image_data, str):
                return None
            
            if image_data.startswith('data:image'):
                header, encoded = image_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                image_bytes = base64.b64decode(image_data)
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                print("❌ Failed to decode image", file=sys.stderr)
                return None
            
            print(f"📷 Decoded image: {image.shape}", file=sys.stderr)
            return image
            
        except Exception as e:
            print(f"❌ Base64 decoding failed: {e}", file=sys.stderr)
            return None
    
    def detect_face_simple(self, image):
        """Simple face detection with fallback"""
        try:
            if image is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Try to load face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                print("⚠️ No face cascade found, using entire image", file=sys.stderr)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(image_rgb, (self.input_shape[1], self.input_shape[0]))
                return resized
            
            face_cascade = cv2.CascadeClassifier(cascade_path)
            if face_cascade.empty():
                print("⚠️ Face cascade empty, using entire image", file=sys.stderr)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(image_rgb, (self.input_shape[1], self.input_shape[0]))
                return resized
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print(f"🤖 Detected {len(faces)} faces", file=sys.stderr)
            
            if len(faces) == 0:
                print("⚠️ No faces detected, using entire image", file=sys.stderr)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(image_rgb, (self.input_shape[1], self.input_shape[0]))
                return resized
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add padding
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Crop and resize
            face = image[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (self.input_shape[1], self.input_shape[0]))
            
            print(f"✅ Cropped face: {face_resized.shape}", file=sys.stderr)
            return face_resized
            
        except Exception as e:
            print(f"❌ Face detection failed: {e}", file=sys.stderr)
            # Fallback
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(image_rgb, (self.input_shape[1], self.input_shape[0]))
            return resized
    
    def preprocess_image(self, image):
        """Prepare image for model"""
        try:
            if image is None:
                return None
            
            # Normalize and ensure correct shape
            image = image.astype('float32') / 255.0
            
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"❌ Preprocessing failed: {e}", file=sys.stderr)
            return None
    
    def verify_faces(self, id_image_input, webcam_image_input):
        """Main verification function"""
        print("\n" + "="*60, file=sys.stderr)
        print("🎯 STARTING SAFE FACE VERIFICATION", file=sys.stderr)
        print("="*60, file=sys.stderr)
        
        try:
            # Check model
            if self.model is None:
                return self.error_result("Model not loaded")
            
            # Load ID image
            print("\n📸 Processing ID Image...", file=sys.stderr)
            id_image = cv2.imread(id_image_input)
            if id_image is None:
                return self.error_result("Failed to load ID image")
            print(f"📐 ID Image shape: {id_image.shape}", file=sys.stderr)
            
            # Load webcam image
            print("\n📹 Processing Webcam Image...", file=sys.stderr)
            webcam_image = cv2.imread(webcam_image_input)
            if webcam_image is None:
                return self.error_result("Failed to load webcam image")
            print(f"📐 Webcam Image shape: {webcam_image.shape}", file=sys.stderr)
            
            # Detect faces
            print("\n🤖 Detecting Faces...", file=sys.stderr)
            id_face = self.detect_face_simple(id_image)
            webcam_face = self.detect_face_simple(webcam_image)
            
            if id_face is None or webcam_face is None:
                return self.error_result("Face detection failed")
            
            # Preprocess
            print("\n⚙️ Preprocessing for Model...", file=sys.stderr)
            id_processed = self.preprocess_image(id_face)
            webcam_processed = self.preprocess_image(webcam_face)
            
            if id_processed is None or webcam_processed is None:
                return self.error_result("Image preprocessing failed")
            
            # Model prediction
            print("\n🧠 Running Model Prediction...", file=sys.stderr)
            similarity_score = self.model.predict(
                [id_processed, webcam_processed], 
                verbose=0
            )[0][0]
            
            print(f"📊 Raw Similarity Score: {similarity_score:.6f}", file=sys.stderr)
            
            # Calculate results with adjusted threshold
            threshold = 0.5  # More lenient threshold
            verified = similarity_score >= threshold
            confidence = float(similarity_score)
            
            result = {
                "verified": bool(verified),
                "confidence": confidence,
                "similarity_score": float(similarity_score),
                "message": "Safe verification completed"
            }
            
            print(f"\n🎉 FINAL RESULT:", file=sys.stderr)
            print(f"   Verified: {verified}", file=sys.stderr)
            print(f"   Confidence: {confidence:.2%}", file=sys.stderr)
            print(f"   Similarity: {similarity_score:.4f}", file=sys.stderr)
            print("="*60, file=sys.stderr)
            
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            print(f"🔍 Traceback: {traceback.format_exc()}", file=sys.stderr)
            return self.error_result(error_msg)
    
    def error_result(self, error_message):
        """Helper method for error results"""
        return {
            "verified": False,
            "confidence": 0.0,
            "error": error_message
        }

def main():
    if len(sys.argv) != 3:
        result = {
            "verified": False, 
            "confidence": 0.0, 
            "error": f"Expected 2 arguments, got {len(sys.argv)-1}"
        }
        print(json.dumps(result))
        return
    
    id_image_input = sys.argv[1]
    webcam_image_input = sys.argv[2]
    
    print(f"🔧 Input Summary:", file=sys.stderr)
    print(f"   ID Image: {id_image_input}", file=sys.stderr)
    print(f"   Webcam Image: {webcam_image_input}", file=sys.stderr)
    
    # Initialize and run verification
    verifier = SafeFaceVerifier()
    result = verifier.verify_faces(id_image_input, webcam_image_input)
    
    # Output result
    print(json.dumps(result))

if __name__ == "__main__":
    main()