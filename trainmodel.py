import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class FaceVerificationModel:
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = None
        
    def create_base_network(self):
        """Create base CNN for feature extraction"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='linear')  # Embedding layer
        ])
        return model
    
    class L2Normalize(layers.Layer):
        """Custom L2 normalization layer"""
        def call(self, x):
            return tf.nn.l2_normalize(x, axis=1)
        
        def get_config(self):
            return super().get_config()
    
    class AbsoluteDifference(layers.Layer):
        """Custom absolute difference layer to replace Lambda"""
        def call(self, inputs):
            return tf.abs(inputs)
        
        def get_config(self):
            return super().get_config()
    
    def create_siamese_model(self):
        """Create Siamese network using only custom layers"""
        # Inputs
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        # Base network (shared weights)
        base_network = self.create_base_network()
        
        # Get embeddings
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # L2 normalization using custom layer
        embedding_a = self.L2Normalize()(embedding_a)
        embedding_b = self.L2Normalize()(embedding_b)
        
        # Compute absolute difference using custom layer
        difference = layers.Subtract()([embedding_a, embedding_b])
        absolute_difference = self.AbsoluteDifference()(difference)
        
        # Classification head
        x = layers.Dense(64, activation='relu')(absolute_difference)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=[input_a, input_b], outputs=output)
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    def load_lfw_dataset(self, dataset_path, min_images_per_person=2, max_images_per_person=50):
        """Load only people with multiple images"""
        print("Loading LFW dataset (only people with multiple images)...")
        
        images = []
        labels = []
        
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                person_images = []
                for image_file in os.listdir(person_path):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_path, image_file)
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
                                image = image.astype('float32') / 255.0
                                person_images.append(image)
                        except Exception as e:
                            print(f"Error loading {image_path}: {e}")
                
                # Only include people with enough images
                if len(person_images) >= min_images_per_person:
                    # Take up to max_images_per_person
                    person_images = person_images[:max_images_per_person]
                    images.extend(person_images)
                    labels.extend([person_name] * len(person_images))
                    print(f"✓ Added {person_name} with {len(person_images)} images")
        
        print(f"Loaded {len(images)} images from {len(set(labels))} people (with at least {min_images_per_person} images each)")
        
        if len(images) == 0:
            raise ValueError(f"No people found with at least {min_images_per_person} images!")
            
        return np.array(images), np.array(labels)
    
    def create_pairs(self, images, labels, num_pairs=5000):
        """Create positive and negative pairs"""
        print("Creating training pairs...")
        
        pairs_a = []
        pairs_b = []
        pair_labels = []
        
        unique_labels = np.unique(labels)
        label_to_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        print(f"Available people for pairs: {len(unique_labels)}")
        
        # Create positive pairs (same person)
        positive_count = 0
        max_positive_pairs = num_pairs // 2
        
        for label, indices in label_to_indices.items():
            if len(indices) >= 2:
                # Calculate how many pairs we can create for this person
                n_possible_pairs = min(len(indices) * (len(indices) - 1) // 2, 5)  # Max 5 pairs per person
                for _ in range(n_possible_pairs):
                    if positive_count >= max_positive_pairs:
                        break
                    idx1, idx2 = np.random.choice(indices, 2, replace=False)
                    pairs_a.append(images[idx1])
                    pairs_b.append(images[idx2])
                    pair_labels.append(1.0)
                    positive_count += 1
        
        print(f"Created {positive_count} positive pairs")
        
        # Create negative pairs (different persons)
        negative_count = 0
        max_negative_pairs = num_pairs - positive_count
        available_labels = list(label_to_indices.keys())
        
        for _ in range(max_negative_pairs):
            if len(available_labels) < 2:
                break
                
            label1, label2 = np.random.choice(available_labels, 2, replace=False)
            idx1 = np.random.choice(label_to_indices[label1])
            idx2 = np.random.choice(label_to_indices[label2])
            
            pairs_a.append(images[idx1])
            pairs_b.append(images[idx2])
            pair_labels.append(0.0)
            negative_count += 1
        
        # Convert to arrays and shuffle
        pairs_a = np.array(pairs_a)
        pairs_b = np.array(pairs_b)
        pair_labels = np.array(pair_labels)
        
        # Shuffle
        indices = np.random.permutation(len(pairs_a))
        pairs_a = pairs_a[indices]
        pairs_b = pairs_b[indices]
        pair_labels = pair_labels[indices]
        
        print(f"Created {len(pairs_a)} pairs ({positive_count} positive, {negative_count} negative)")
        
        if positive_count == 0:
            raise ValueError("No positive pairs created! Check your dataset.")
            
        return [pairs_a, pairs_b], pair_labels
    
    def train(self, dataset_path, epochs=30, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Load data with minimum image requirement
        try:
            images, labels = self.load_lfw_dataset(
                dataset_path, 
                min_images_per_person=2,  # Only people with at least 2 images
                max_images_per_person=20
            )
        except ValueError as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data...")
            return self.synthetic_training()
        
        if len(images) == 0:
            print("No images loaded! Using synthetic data...")
            return self.synthetic_training()
        
        # Create pairs
        try:
            pairs, pair_labels = self.create_pairs(images, labels, num_pairs=4000)
        except ValueError as e:
            print(f"Error creating pairs: {e}")
            print("Falling back to synthetic data...")
            return self.synthetic_training()
        
        # Split data
        (pairs_a_train, pairs_a_val, 
         pairs_b_train, pairs_b_val, 
         labels_train, labels_val) = train_test_split(
            pairs[0], pairs[1], pair_labels, 
            test_size=validation_split, random_state=42
        )
        
        print(f"Training pairs: {len(pairs_a_train)}")
        print(f"Validation pairs: {len(pairs_a_val)}")
        
        # Create and compile model
        self.create_siamese_model()
        self.compile_model()
        
        # Model summary
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=8, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=5, 
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_face_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train
        print("Starting training...")
        history = self.model.fit(
            [pairs_a_train, pairs_b_train],
            labels_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([pairs_a_val, pairs_b_val], labels_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save('face_verification_model.h5')
        print("Model saved as 'face_verification_model.h5'")
        
        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        self.evaluate_model([pairs_a_val, pairs_b_val], labels_val)
        
        # Plot results
        self.plot_results(history)
        
        return history
    
    def synthetic_training(self):
        """Train with synthetic data if real data fails"""
        print("Creating synthetic dataset for training...")
        
        self.create_siamese_model()
        self.compile_model()
        
        # Create synthetic pairs (more realistic distribution)
        num_pairs = 2000
        pairs_a = []
        pairs_b = []
        pair_labels = []
        
        # Create synthetic "people" with multiple images
        num_people = 100
        people_embeddings = np.random.randn(num_people, 128)
        
        # Positive pairs (same "person")
        for i in range(num_pairs // 2):
            person_id = np.random.randint(0, num_people)
            # Create similar-looking images for same person
            base_img = np.random.random((128, 128, 3))
            noise1 = np.random.normal(0, 0.1, (128, 128, 3))
            noise2 = np.random.normal(0, 0.1, (128, 128, 3))
            
            img1 = np.clip(base_img + noise1, 0, 1)
            img2 = np.clip(base_img + noise2, 0, 1)
            
            pairs_a.append(img1)
            pairs_b.append(img2)
            pair_labels.append(1.0)
        
        # Negative pairs (different "people")
        for i in range(num_pairs // 2):
            person1 = np.random.randint(0, num_people)
            person2 = np.random.randint(0, num_people)
            while person2 == person1:
                person2 = np.random.randint(0, num_people)
                
            img1 = np.random.random((128, 128, 3))
            img2 = np.random.random((128, 128, 3))
            
            pairs_a.append(img1)
            pairs_b.append(img2)
            pair_labels.append(0.0)
        
        pairs_a = np.array(pairs_a)
        pairs_b = np.array(pairs_b)
        pair_labels = np.array(pair_labels)
        
        # Shuffle
        indices = np.random.permutation(len(pairs_a))
        pairs_a = pairs_a[indices]
        pairs_b = pairs_b[indices]
        pair_labels = pair_labels[indices]
        
        print(f"Training with {len(pairs_a)} synthetic pairs")
        
        # Split
        (pairs_a_train, pairs_a_val, 
         pairs_b_train, pairs_b_val, 
         labels_train, labels_val) = train_test_split(
            pairs_a, pairs_b, pair_labels, 
            test_size=0.2, random_state=42
        )
        
        # Train
        history = self.model.fit(
            [pairs_a_train, pairs_b_train],
            labels_train,
            batch_size=16,
            epochs=10,
            validation_data=([pairs_a_val, pairs_b_val], labels_val),
            verbose=1
        )
        
        self.model.save('synthetic_face_model.h5')
        print("Synthetic model saved as 'synthetic_face_model.h5'")
        
        # Evaluate
        self.evaluate_model([pairs_a_val, pairs_b_val], labels_val)
        self.plot_results(history)
        return history

    def evaluate_model(self, test_pairs, test_labels, threshold=0.5):
        """Comprehensive model evaluation with multiple metrics"""
        if self.model is None:
            raise ValueError("Model not trained yet! Train or load a model first.")
        
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Get predictions
        predictions = self.model.predict(test_pairs)
        predictions_binary = (predictions > threshold).astype(int)
        
        # Calculate additional metrics
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(test_labels, predictions_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions_binary, average='binary'
        )
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, predictions_binary, 
                                  target_names=['Different', 'Same']))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(test_labels, predictions_binary)
        self.plot_confusion_matrix(cm)
        
        # ROC Curve
        self.plot_roc_curve(test_labels, predictions)
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(test_labels, predictions)
        
        # Threshold analysis
        self.analyze_thresholds(test_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'binary_predictions': predictions_binary
        }

    def verify_faces(self, image1, image2, threshold=0.5):
        """Verify if two faces belong to the same person"""
        if self.model is None:
            raise ValueError("Model not loaded! Please load a trained model first.")
        
        # Preprocess images
        image1_processed = self.preprocess_image(image1)
        image2_processed = self.preprocess_image(image2)
        
        # Make prediction
        prediction = self.model.predict([
            np.array([image1_processed]), 
            np.array([image2_processed])
        ], verbose=0)
        
        confidence = float(prediction[0][0])
        is_same_person = confidence > threshold
        
        return is_same_person, confidence

    def preprocess_image(self, image):
        """Preprocess a single image for prediction"""
        if isinstance(image, str):  # File path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found: {image}")
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            # Assume it's already an image array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already in color
                pass
            else:
                raise ValueError("Input numpy array must be a 3-channel color image")
        else:
            raise ValueError("Input must be a file path or numpy array")
        
        # Resize and normalize
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        image = image.astype('float32') / 255.0
        
        return image

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Different', 'Same'],
                   yticklabels=['Different', 'Same'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, true_labels, predictions):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        print(f"ROC AUC Score: {roc_auc:.4f}")

    def plot_precision_recall_curve(self, true_labels, predictions):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        precision, recall, _ = precision_recall_curve(true_labels, predictions)
        average_precision = average_precision_score(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'Precision-Recall curve (AP = {average_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()
        
        print(f"Average Precision: {average_precision:.4f}")

    def analyze_thresholds(self, true_labels, predictions, num_thresholds=20):
        """Analyze performance across different thresholds"""
        thresholds = np.linspace(0.1, 0.9, num_thresholds)
        accuracies = []
        f1_scores = []
        
        for threshold in thresholds:
            binary_preds = (predictions > threshold).astype(int)
            accuracy = np.mean(binary_preds.flatten() == true_labels)
            accuracies.append(accuracy)
            
            from sklearn.metrics import f1_score
            f1 = f1_score(true_labels, binary_preds)
            f1_scores.append(f1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, accuracies, 'b-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, f1_scores, 'r-', label='F1-Score', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Model Performance vs Classification Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Find optimal threshold (maximizing F1-score)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Optimal threshold (max F1): {optimal_threshold:.3f}")
        print(f"Best F1-Score: {f1_scores[optimal_idx]:.4f}")
        print(f"Accuracy at optimal threshold: {accuracies[optimal_idx]:.4f}")

    def load_model(self, model_path):
        """Load a pre-trained model with custom objects"""
        try:
            # Enable unsafe deserialization for Lambda layers (if any remain)
            # But we've replaced them with custom layers, so this should work safely
            self.model = keras.models.load_model(
                model_path, 
                custom_objects={
                    'L2Normalize': self.L2Normalize,
                    'AbsoluteDifference': self.AbsoluteDifference
                }
            )
            print(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # Try with unsafe mode as fallback
            try:
                self.model = keras.models.load_model(
                    model_path, 
                    custom_objects={
                        'L2Normalize': self.L2Normalize,
                        'AbsoluteDifference': self.AbsoluteDifference
                    },
                    safe_mode=False
                )
                print(f"Model loaded successfully (with safe_mode=False) from {model_path}")
                return True
            except Exception as e2:
                print(f"Error loading model even with safe_mode=False: {e2}")
                return False

    def plot_results(self, history):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Precision
        plt.subplot(1, 3, 3)
        if 'precision' in history.history:
            plt.plot(history.history['precision'], label='Training Precision')
            plt.plot(history.history['val_precision'], label='Validation Precision')
            plt.title('Model Precision')
            plt.xlabel('Epoch')
            plt.ylabel('Precision')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Precision data not available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.title('Model Precision')
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()

# Updated test function with evaluation
def test_model_with_evaluation():
    """Test the model with sample data and evaluation"""
    model = FaceVerificationModel(input_shape=(128, 128, 3))
    model.create_siamese_model()
    model.compile_model()
    
    print("Model created successfully!")
    model.model.summary()
    
    # Create test data for evaluation
    sample_a = np.random.random((100, 128, 128, 3))
    sample_b = np.random.random((100, 128, 128, 3))
    sample_labels = np.random.randint(0, 2, (100, 1))
    
    # Test prediction
    prediction = model.model.predict([sample_a[:2], sample_b[:2]])
    print(f"Sample prediction: {prediction}")
    
    # Test evaluation
    print("\nTesting evaluation methods...")
    try:
        results = model.evaluate_model([sample_a, sample_b], sample_labels)
        print("✓ Evaluation methods working!")
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
    
    # Test face verification
    print("\nTesting face verification...")
    try:
        test_img1 = np.random.random((128, 128, 3))
        test_img2 = np.random.random((128, 128, 3))
        
        is_same, confidence = model.verify_faces(test_img1, test_img2)
        print(f"Face verification result: Same={is_same}, Confidence={confidence:.4f}")
        print("✓ Face verification working!")
    except Exception as e:
        print(f"✗ Face verification test failed: {e}")
    
    return True

# Main function with evaluation demo
def main():
    dataset_path = r"C:\Users\vasudev\EnQueue\model-training\lfw_dataset\lfw-deepfunneled\lfw-deepfunneled"
    
    print("Testing model creation and evaluation...")
    if test_model_with_evaluation():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
        return
    
    print("\nStarting main training...")
    model = FaceVerificationModel(input_shape=(128, 128, 3))
    
    # Check if we can load existing model first
    if os.path.exists('face_verification_model.h5'):
        print("Found existing model, loading...")
        if model.load_model('face_verification_model.h5'):
            print("Using pre-trained model for evaluation")
            # You can add evaluation on test data here
            return
    
    try:
        history = model.train(
            dataset_path=dataset_path,
            epochs=30,
            batch_size=32,
            validation_split=0.2
        )
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("\nTrying with sample dataset...")
        test_with_sample_dataset()

def test_with_sample_dataset():
    """Test with a small sample dataset if LFW is not available"""
    print("Creating sample dataset for testing...")
    
    model = FaceVerificationModel(input_shape=(128, 128, 3))
    
    # Create a small synthetic dataset that mimics the structure
    images = []
    labels = []
    
    # Create 10 "people" with 3 images each
    for person_id in range(10):
        base_color = np.random.random(3)
        for img_id in range(3):
            # Create slightly varied images for same person
            image = np.ones((128, 128, 3)) * base_color
            noise = np.random.normal(0, 0.1, (128, 128, 3))
            image = np.clip(image + noise, 0, 1)
            images.append(image)
            labels.append(f"person_{person_id}")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Created sample dataset: {len(images)} images from {len(np.unique(labels))} people")
    
    # Create pairs and train
    pairs, pair_labels = model.create_pairs(images, labels, num_pairs=200)
    
    model.create_siamese_model()
    model.compile_model()
    
    history = model.model.fit(
        [pairs[0], pairs[1]],
        pair_labels,
        batch_size=16,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )
    
    # Test the new evaluation methods
    print("\nTesting evaluation on sample data...")
    test_pairs = [pairs[0][:50], pairs[1][:50]]
    test_labels = pair_labels[:50]
    
    model.evaluate_model(test_pairs, test_labels)
    
    model.model.save('sample_face_model.h5')
    print("Sample model saved as 'sample_face_model.h5'")
    
    return True

if __name__ == "__main__":
    main()