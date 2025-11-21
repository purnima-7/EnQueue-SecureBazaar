import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ------------------------------------------
#  Fix for model loading (Lambda layer issue)
# ------------------------------------------
keras.config.enable_unsafe_deserialization()

# ------------------------------------------
#  CUSTOM LAYERS (same as training)
# ------------------------------------------
class L2Normalize(keras.layers.Layer):
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=1)

class AbsoluteDifference(keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)

# ------------------------------------------
#  FACE VERIFICATION MODEL (Inference Only)
# ------------------------------------------
class FaceVerificationModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None

    def load_model(self, model_path):
        """Load the trained Siamese model"""
        try:
            self.model = keras.models.load_model(
                model_path,
                custom_objects={
                    'L2Normalize': L2Normalize,
                    'AbsoluteDifference': AbsoluteDifference
                },
                safe_mode=False
            )
            print(f"✅ Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def preprocess_image(self, image_path):
        """Reads and preprocesses a face image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error loading image: {image_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = img.astype("float32") / 255.0

        return img

    def verify(self, img1_path, img2_path, threshold=0.5):
        """Compares two images and returns confidence & decision"""
        try:
            img1 = self.preprocess_image(img1_path)
            img2 = self.preprocess_image(img2_path)

            pred = self.model.predict([
                np.array([img1]),
                np.array([img2])
            ], verbose=0)

            confidence = float(pred[0][0])
            is_same = confidence >= threshold

            return is_same, confidence
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False, 0.0

# ------------------------------------------
#  ENHANCED MAIN FUNCTION
# ------------------------------------------
def main():
    # Configuration
    model_path = r"face_verification_model.h5"
    img1_path = r"C:\Users\vasudev\Downloads\WhatsApp Image 2025-11-13 at 04.28.23_bc653052.jpg"
    img2_path = r"C:\Users\vasudev\Downloads\WhatsApp Image 2025-11-11 at 22.09.20_1132becc.jpg"
    # Initialize model
    fv = FaceVerificationModel()

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available model files:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
        return

    # Load model
    if not fv.load_model(model_path):
        print("❌ Failed to load model. Trying alternative models...")
        
        # Try alternative model files
        alternative_models = [
            'best_face_model.h5',
            'synthetic_face_model.h5', 
            'sample_face_model.h5',
            'improved_face_verification_model.h5'
        ]
        
        for alt_model in alternative_models:
            if os.path.exists(alt_model):
                print(f"🔄 Trying alternative model: {alt_model}")
                if fv.load_model(alt_model):
                    break
        else:
            print("❌ No alternative models worked.")
            return

    # Check if image files exist
    for img_path in [img1_path, img2_path]:
        if not os.path.exists(img_path):
            print(f"❌ Image file not found: {img_path}")
            return

    print("\n🔍 Comparing:")
    print(f"Image 1: {os.path.basename(img1_path)}")
    print(f"Image 2: {os.path.basename(img2_path)}")

    # Perform verification
    same, confidence = fv.verify(img1_path, img2_path, threshold=0.5)

    print("\n📊 RESULT:")
    print(f"Same Person: {same}")
    print(f"Confidence Score: {confidence:.4f}")
    
    # Interpretation
    if confidence > 0.7:
        print("🎯 Strong match!")
    elif confidence > 0.5:
        print("✅ Likely match")
    elif confidence > 0.3:
        print("❓ Uncertain")
    else:
        print("❌ Different persons")

if __name__ == "__main__":
    main()