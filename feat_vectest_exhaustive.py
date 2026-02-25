import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from itertools import combinations
import time
from collections import defaultdict

class FeatureExtractorEvaluator:
    def __init__(self, model_path, input_tensor_name="images:0", output_tensor_name="features:0"):
        """Initialize evaluator with a .pb model"""
        self.model_path = model_path
        self.graph = self._load_model()
        self.input_tensor = self.graph.get_tensor_by_name(input_tensor_name)
        self.output_tensor = self.graph.get_tensor_by_name(output_tensor_name)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        
    def _load_model(self):
        """Load frozen .pb model"""
        with tf.io.gfile.GFile(self.model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        return graph
    
    def preprocess_image(self, image_path, target_size=(64, 128)):  #128, 64 for mobnet / 64, 128 otherwise
        """Preprocess image for model"""
        img = cv2.imread(image_path)  # Keep BGR
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img = cv2.resize(img, target_size)  # width, height

        img = img.astype(np.float32)    # ✅ must match training
        return np.expand_dims(img, axis=0)
    
    def get_embedding(self, image_path):
        """Extract embedding for single image"""
        img = self.preprocess_image(image_path)
        embedding = self.sess.run(self.output_tensor, {self.input_tensor: img})[0]
        return embedding
    
    def cosine_similarity(self, emb1, emb2):
        """Compute cosine similarity"""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def exhaustive_pairwise_test(self, dataset_path, max_pairs_per_class=100, test_ratio=0.3):
        """
        Perform exhaustive pairwise testing across all classes
        
        Args:
            dataset_path: Path to dataset with folder structure
            max_pairs_per_class: Maximum pairs to test per class (for speed)
            test_ratio: Ratio of images to use for testing (rest for training simulation)
        """
        # Load all image paths
        class_folders = sorted([d for d in os.listdir(dataset_path) 
                              if os.path.isdir(os.path.join(dataset_path, d))])
        
        print(f"Found {len(class_folders)} classes")
        
        # Collect all image paths per class
        class_images = {}
        for class_folder in class_folders:
            folder_path = os.path.join(dataset_path, class_folder)
            images = [os.path.join(folder_path, f) 
                     for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(images) >= 2:  # Need at least 2 images per class
                class_images[class_folder] = images
                print(f"  {class_folder}: {len(images)} images")
        
        # Generate test pairs
        positive_scores = []
        negative_scores = []
        positive_pairs = []
        negative_pairs = []
        
        print("\nGenerating test pairs...")
        
        # Positive pairs (same class)
        for class_name, images in class_images.items():
            if len(images) < 2:
                continue
                
            # Split into train/test (simulate real usage)
            split_idx = int(len(images) * (1 - test_ratio))
            test_images = images[split_idx:] if split_idx < len(images) else images[-2:]
            
            # Generate all combinations within test set
            pairs = list(combinations(test_images, 2))
            
            # Limit number of pairs for speed
            if len(pairs) > max_pairs_per_class:
                indices = np.random.choice(len(pairs), max_pairs_per_class, replace=False)
                pairs = [pairs[i] for i in indices]
            
            for img1, img2 in pairs:
                positive_pairs.append((img1, img2))
        
        # Negative pairs (different classes)
        class_names = list(class_images.keys())
        for i, class1 in enumerate(class_names):
            for class2 in class_names[i+1:]:
                images1 = class_images[class1]
                images2 = class_images[class2]
                
                # Split into train/test
                split1 = int(len(images1) * (1 - test_ratio))
                split2 = int(len(images2) * (1 - test_ratio))
                test1 = images1[split1:] if split1 < len(images1) else images1[-1:]
                test2 = images2[split2:] if split2 < len(images2) else images2[-1:]
                
                # Generate pairs between classes
                max_pairs = min(max_pairs_per_class // len(class_names), len(test1) * len(test2))
                pairs_generated = 0
                
                for img1 in test1:
                    for img2 in test2:
                        negative_pairs.append((img1, img2))
                        pairs_generated += 1
                        if pairs_generated >= max_pairs:
                            break
                    if pairs_generated >= max_pairs:
                        break
        
        print(f"Generated {len(positive_pairs)} positive pairs")
        print(f"Generated {len(negative_pairs)} negative pairs")
        
        # Compute similarities
        print("\nComputing similarities...")
        start_time = time.time()
        
        # Positive pairs
        for i, (img1, img2) in enumerate(positive_pairs):
            try:
                emb1 = self.get_embedding(img1)
                emb2 = self.get_embedding(img2)
                sim = self.cosine_similarity(emb1, emb2)
                positive_scores.append(sim)
                
                if i % 100 == 0:
                    print(f"  Positive pairs processed: {i}/{len(positive_pairs)}")
                    
            except Exception as e:
                print(f"Error processing {img1}, {img2}: {e}")
        
        # Negative pairs
        for i, (img1, img2) in enumerate(negative_pairs):
            try:
                emb1 = self.get_embedding(img1)
                emb2 = self.get_embedding(img2)
                sim = self.cosine_similarity(emb1, emb2)
                negative_scores.append(sim)
                
                if i % 100 == 0:
                    print(f"  Negative pairs processed: {i}/{len(negative_pairs)}")
                    
            except Exception as e:
                print(f"Error processing {img1}, {img2}: {e}")
        
        elapsed = time.time() - start_time
        print(f"\nComputation time: {elapsed:.2f} seconds")
        
        # Convert to arrays
        positive_scores = np.array(positive_scores)
        negative_scores = np.array(negative_scores)
        
        return positive_scores, negative_scores, positive_pairs, negative_pairs
    
    def compute_metrics(self, positive_scores, negative_scores):
        """Compute comprehensive evaluation metrics"""
        
        # Basic statistics
        metrics = {
            'positive_mean': np.mean(positive_scores),
            'positive_std': np.std(positive_scores),
            'positive_min': np.min(positive_scores),
            'positive_max': np.max(positive_scores),
            'negative_mean': np.mean(negative_scores),
            'negative_std': np.std(negative_scores),
            'negative_min': np.min(negative_scores),
            'negative_max': np.max(negative_scores),
        }
        
        # Create labels and scores for ROC/AUC
        y_true = np.concatenate([np.ones_like(positive_scores), 
                                 np.zeros_like(negative_scores)])
        y_scores = np.concatenate([positive_scores, negative_scores])
        
        # ROC Curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        metrics['roc_auc'] = roc_auc
        
        # Find optimal threshold (maximizing Youden's J statistic)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        metrics['optimal_threshold'] = optimal_threshold
        metrics['optimal_tpr'] = tpr[optimal_idx]
        metrics['optimal_fpr'] = fpr[optimal_idx]
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        metrics['pr_auc'] = pr_auc
        
        # Accuracy at optimal threshold
        predictions = (y_scores > optimal_threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        metrics['accuracy'] = accuracy
        
        # F1 score
        tp = np.sum((predictions == 1) & (y_true == 1))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score) \
            if (precision_score + recall_score) > 0 else 0
        
        metrics['precision'] = precision_score
        metrics['recall'] = recall_score
        metrics['f1_score'] = f1
        
        # Separation metrics
        metrics['separation'] = metrics['positive_mean'] - metrics['negative_mean']
        metrics['separability_index'] = metrics['separation'] / np.sqrt(
            metrics['positive_std']**2 + metrics['negative_std']**2)
        
        return metrics, fpr, tpr, precision, recall
    
    def plot_results(self, positive_scores, negative_scores, metrics, 
                    fpr=None, tpr=None, precision=None, recall=None):
        """Plot comprehensive evaluation results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Distribution plot
        axes[0, 0].hist(positive_scores, alpha=0.5, label='Positive', bins=30, density=True)
        axes[0, 0].hist(negative_scores, alpha=0.5, label='Negative', bins=30, density=True)
        axes[0, 0].axvline(metrics['optimal_threshold'], color='red', 
                          linestyle='--', label=f"Threshold: {metrics['optimal_threshold']:.3f}")
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Score Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC Curve
        if fpr is not None and tpr is not None:
            axes[0, 1].plot(fpr, tpr, 'b-', label=f'AUC = {metrics["roc_auc"]:.3f}')
            axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
            axes[0, 1].scatter(metrics['optimal_fpr'], metrics['optimal_tpr'], 
                              color='red', s=100, 
                              label=f'Optimal (FPR={metrics["optimal_fpr"]:.3f})')
            axes[0, 1].set_xlabel('False Positive Rate')
            axes[0, 1].set_ylabel('True Positive Rate')
            axes[0, 1].set_title('ROC Curve')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axis('equal')
        
        # 3. Precision-Recall Curve
        if precision is not None and recall is not None:
            axes[0, 2].plot(recall, precision, 'g-', label=f'AUC = {metrics["pr_auc"]:.3f}')
            axes[0, 2].set_xlabel('Recall')
            axes[0, 2].set_ylabel('Precision')
            axes[0, 2].set_title('Precision-Recall Curve')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot
        axes[1, 0].boxplot([positive_scores, negative_scores], 
                          labels=['Positive', 'Negative'])
        axes[1, 0].set_ylabel('Cosine Similarity')
        axes[1, 0].set_title('Score Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Metrics table
        metrics_text = f"""
        ROC AUC: {metrics['roc_auc']:.3f}
        PR AUC: {metrics['pr_auc']:.3f}
        Accuracy: {metrics['accuracy']:.3f}
        F1 Score: {metrics['f1_score']:.3f}
        Precision: {metrics['precision']:.3f}
        Recall: {metrics['recall']:.3f}
        Optimal Threshold: {metrics['optimal_threshold']:.3f}
        Separation: {metrics['separation']:.3f}
        Separability Index: {metrics['separability_index']:.3f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, 
                       verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        # 6. Confusion matrix (simulated)
        n_pos = len(positive_scores)
        n_neg = len(negative_scores)
        tp = np.sum(positive_scores > metrics['optimal_threshold'])
        tn = np.sum(negative_scores <= metrics['optimal_threshold'])
        fp = n_neg - tn
        fn = n_pos - tp
        
        cm = np.array([[tp, fn], [fp, tn]])
        im = axes[1, 2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 2].set_title('Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 2].text(j, i, str(cm[i, j]), 
                               ha="center", va="center", 
                               color="white" if cm[i, j] > cm.max()/2 else "black")
        
        axes[1, 2].set_xticks([0, 1])
        axes[1, 2].set_yticks([0, 1])
        axes[1, 2].set_xticklabels(['Pred Pos', 'Pred Neg'])
        axes[1, 2].set_yticklabels(['True Pos', 'True Neg'])
        
        plt.tight_layout()
        plt.savefig("evaluation_results.png")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Positive pairs: {len(positive_scores)}")
        print(f"Negative pairs: {len(negative_scores)}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Optimal threshold: {metrics['optimal_threshold']:.4f}")
        print(f"Separation (Δμ): {metrics['separation']:.4f}")
        print(f"Positive scores: μ={metrics['positive_mean']:.4f}, σ={metrics['positive_std']:.4f}")
        print(f"Negative scores: μ={metrics['negative_mean']:.4f}, σ={metrics['negative_std']:.4f}")
        print("="*60)

# Main execution
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "./runs/turkey_reid/best_model.pb" #"./runs/turkey_reid/best_model.pb" #"./output/frozen_model.pb" #"./output/frozen_model_cnn_highstep.pb"  # Your model "./model_feature_extractor/mars-small128.pb" "./model_feature_extractor/frozen_model.pb" "./output/frozen_model_mobnetv1.pb"
    DATASET_PATH = "./dataset_siam_21"  # Your dataset path
    INPUT_TENSOR = "images:0"  # Update based on your model
    OUTPUT_TENSOR = "features:0" # Update based on your model "Identity:0" "features:0"
    
    # Create evaluator
    print("Loading model...")
    evaluator = FeatureExtractorEvaluator(MODEL_PATH, INPUT_TENSOR, OUTPUT_TENSOR)
    
    # Run exhaustive test
    print("Running exhaustive pairwise test...")
    positive_scores, negative_scores, pos_pairs, neg_pairs = \
        evaluator.exhaustive_pairwise_test(DATASET_PATH, max_pairs_per_class=50)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics, fpr, tpr, precision, recall = \
        evaluator.compute_metrics(positive_scores, negative_scores)
    
    # Plot results
    evaluator.plot_results(positive_scores, negative_scores, metrics, 
                          fpr, tpr, precision, recall)
    
    # Optional: Save results
    results = {
        'positive_scores': positive_scores,
        'negative_scores': negative_scores,
        'metrics': metrics,
        'model_path': MODEL_PATH,
        'dataset_path': DATASET_PATH
    }
    np.savez('evaluation_results.npz', **results)
    print("Results saved to evaluation_results.npz")