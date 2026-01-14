"""
Train All Models Script
This script trains all 6 classification models for the Bank Marketing dataset
Author: BITS Pilani Student
"""

import os
import sys
import time
from importlib import import_module

def main():
    """
    Train all models sequentially
    """
    models = [
        ('Logistic Regression', 'models.logistic_regression'),
        ('Decision Tree', 'models.decision_tree'),
        ('K-Nearest Neighbors', 'models.knn'),
        ('Naive Bayes', 'models.naive_bayes'),
        ('Random Forest', 'models.random_forest'),
        ('XGBoost', 'models.xgboost_model')
    ]

    print("=" * 70)
    print(" " * 15 + "TRAINING ALL CLASSIFICATION MODELS")
    print(" " * 20 + "Bank Marketing Dataset")
    print("=" * 70)
    print(f"\nTotal models to train: {len(models)}\n")

    all_metrics = {}
    total_start_time = time.time()

    for idx, (model_name, module_name) in enumerate(models, 1):
        print(f"\n{'#' * 70}")
        print(f"  [{idx}/{len(models)}] Training: {model_name}")
        print(f"{'#' * 70}\n")

        start_time = time.time()

        try:
            # Import and run the model's main function
            model_module = import_module(module_name)
            metrics = model_module.main()
            all_metrics[model_name] = metrics

            elapsed_time = time.time() - start_time
            print(f"\n✅ {model_name} completed in {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"\n❌ Error training {model_name}: {str(e)}")
            elapsed_time = time.time() - start_time
            print(f"Failed after {elapsed_time:.2f} seconds")

        print(f"\n{'=' * 70}\n")
        time.sleep(1)  # Brief pause between models

    total_elapsed_time = time.time() - total_start_time

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nTotal training time: {total_elapsed_time:.2f} seconds")
    print(f"Models trained successfully: {len(all_metrics)}/{len(models)}\n")

    if all_metrics:
        print("=" * 70)
        print(" " * 20 + "MODEL COMPARISON")
        print("=" * 70)
        print(f"\n{'Model':<25} {'Accuracy':<12} {'AUC':<12} {'F1 Score':<12}")
        print("-" * 70)

        for model_name, metrics in all_metrics.items():
            print(f"{model_name:<25} {metrics['Accuracy']:<12.4f} "
                  f"{metrics['AUC']:<12.4f} {metrics['F1 Score']:<12.4f}")

        print("=" * 70)

        # Find best model
        best_model = max(all_metrics.items(), key=lambda x: x[1]['Accuracy'])
        print(f"\n🏆 Best Model (by Accuracy): {best_model[0]} "
              f"({best_model[1]['Accuracy']:.4f})")

        best_auc_model = max(all_metrics.items(), key=lambda x: x[1]['AUC'])
        print(f"🏆 Best Model (by AUC): {best_auc_model[0]} "
              f"({best_auc_model[1]['AUC']:.4f})")

    print("\n" + "=" * 70)
    print("\n✅ All models have been trained and saved!")
    print("\n📁 Model files saved in: ./models/")
    print("\n🚀 Next step: Run the Streamlit app using:")
    print("   streamlit run app.py")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
