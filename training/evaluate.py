"""
Step 5: Model Evaluation
=========================
Evaluation metrics, confusion matrix, classification report.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    'model_path': '../model/acne_model.h5',
    'test_data_path': '../dataset/test',
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,
    'class_names': ['clear_skin', 'mild', 'moderate', 'severe']
}

def load_trained_model():
    """Load the trained model from disk."""
    print(f"[INFO] Loading model from {CONFIG['model_path']}")
    model = keras.models.load_model(CONFIG['model_path'])
    return model

def create_test_generator():
    """Create test data generator."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        CONFIG['test_data_path'],
        target_size=(CONFIG['img_height'], CONFIG['img_width']),
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        classes=CONFIG['class_names'],
        shuffle=False  # Important for correct label ordering
    )
    
    return test_generator

def evaluate_model(model, test_generator):
    """
    Evaluate model and return metrics.
    """
    print("\n[EVALUATION] Running model evaluation on test set...")
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Get true labels
    true_classes = test_generator.classes
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\n[Test Accuracy]: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n[Classification Report]")
    print("=" * 60)
    report = classification_report(
        true_classes, 
        predicted_classes,
        target_names=CONFIG['class_names'],
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    return accuracy, report, cm, predictions, true_classes

def plot_confusion_matrix(cm, save_path='../static/images/confusion_matrix.png'):
    """
    Plot and save confusion matrix heatmap.
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=CONFIG['class_names'],
        yticklabels=CONFIG['class_names'],
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix - Acne Severity Classification', fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add raw counts as text
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j + 0.5, i + 0.7, f'n={cm[i, j]}', 
                    ha='center', va='center', fontsize=9, color='red')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Confusion matrix saved to {save_path}")

def plot_class_accuracy(predictions, true_classes, save_path='../static/images/class_accuracy.png'):
    """
    Plot per-class accuracy bar chart.
    """
    class_correct = {name: 0 for name in CONFIG['class_names']}
    class_total = {name: 0 for name in CONFIG['class_names']}
    
    pred_classes = np.argmax(predictions, axis=1)
    
    for true, pred in zip(true_classes, pred_classes):
        class_name = CONFIG['class_names'][true]
        class_total[class_name] += 1
        if true == pred:
            class_correct[class_name] += 1
    
    accuracies = [class_correct[name] / class_total[name] * 100 
                  if class_total[name] > 0 else 0 
                  for name in CONFIG['class_names']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CONFIG['class_names'], accuracies, 
                   color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
    
    plt.title('Per-Class Accuracy', fontsize=14, pad=20)
    plt.xlabel('Acne Severity Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Class accuracy plot saved to {save_path}")

def evaluate_full_pipeline():
    """
    Run complete evaluation pipeline.
    """
    print("=" * 60)
    print("ACNEVISION MODEL EVALUATION")
    print("=" * 60)
    
    # Load model
    model = load_trained_model()
    
    # Create test generator
    test_gen = create_test_generator()
    
    # Evaluate
    accuracy, report, cm, predictions, true_classes = evaluate_model(model, test_gen)
    
    # Plot visualizations
    print("\n[STEP 1] Generating confusion matrix...")
    plot_confusion_matrix(cm)
    
    print("[STEP 2] Generating class accuracy chart...")
    plot_class_accuracy(predictions, true_classes)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'predictions': predictions,
        'true_classes': true_classes
    }

if __name__ == '__main__':
    results = evaluate_full_pipeline()