#!/usr/bin/env python3
"""
MoonBoard Model Training Script
Trains an AutoKeras ImageRegressor on preprocessed MoonBoard climbing data.
"""

import numpy as np
import argparse
import os
import pickle
import autokeras as ak
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from moonboard_data_prep import prepare_moonboard_data


def load_prepared_data(data_dir):
    """Load preprocessed data from saved numpy arrays"""
    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Loaded training data: {x_train.shape[0]} samples")
    print(f"Loaded test data: {x_test.shape[0]} samples")
    print(f"Feature shape: {x_train.shape[1:]}")
    
    return x_train, x_test, y_train, y_test


def train_autokeras_model(x_train, y_train, x_test, y_test, 
                         max_trials=1, epochs=3, model_dir="./moonboard_model"):
    """Train AutoKeras ImageRegressor model"""
    
    print("Initializing AutoKeras ImageRegressor...")
    reg = ak.ImageRegressor(
        overwrite=True, 
        max_trials=max_trials,
        directory=model_dir
    )
    
    print(f"Training model with {max_trials} trial(s) and {epochs} epochs...")
    reg.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    
    return reg


def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model"""
    print("\nEvaluating model...")
    
    # Get predictions
    predicted_y = model.predict(x_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predicted_y)
    mae = mean_absolute_error(y_test, predicted_y)
    
    # AutoKeras evaluation
    ak_evaluation = model.evaluate(x_test, y_test)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"AutoKeras Evaluation: {ak_evaluation}")
    
    return predicted_y, {"mse": mse, "mae": mae, "ak_eval": ak_evaluation}


def save_model_and_results(model, predictions, metrics, y_test, output_dir):
    """Save the trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model (AutoKeras handles this automatically in the model directory)
    print(f"Model saved in AutoKeras directory")
    
    # Save predictions and actual values
    np.save(os.path.join(output_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"Results saved to {output_dir}")


def plot_results(predictions, y_test, output_dir=None):
    """Plot prediction vs actual results"""
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    plt.title('Predicted vs Actual Grades')
    plt.grid(True, alpha=0.3)
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    residuals = predictions.flatten() - y_test
    plt.scatter(y_test, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Grade')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'model_results.png'), dpi=300, bbox_inches='tight')
        print(f"Plot saved to {os.path.join(output_dir, 'model_results.png')}")
    
    plt.show()


def train_moonboard_model(json_file=None, data_dir=None, max_trials=1, epochs=3, 
                         test_size=1000, random_state=42, output_dir="./model_output",
                         model_dir="./moonboard_model", plot=True):
    """
    Complete model training pipeline
    
    Args:
        json_file (str): Path to JSON file (if preparing data from scratch)
        data_dir (str): Directory with preprocessed data (alternative to json_file)
        max_trials (int): Maximum AutoKeras trials
        epochs (int): Training epochs
        test_size (int): Test set size (only used if preparing from JSON)
        random_state (int): Random seed
        output_dir (str): Directory to save results
        model_dir (str): Directory for AutoKeras model
        plot (bool): Whether to generate plots
    
    Returns:
        tuple: (model, predictions, metrics)
    """
    
    # Load or prepare data
    if data_dir and os.path.exists(data_dir):
        print(f"Loading preprocessed data from {data_dir}...")
        x_train, x_test, y_train, y_test = load_prepared_data(data_dir)
    elif json_file and os.path.exists(json_file):
        print(f"Preparing data from {json_file}...")
        x_train, x_test, y_train, y_test, _ = prepare_moonboard_data(
            json_file, test_size=test_size, random_state=random_state
        )
    else:
        raise ValueError("Must provide either data_dir with preprocessed data or json_file")
    
    # Additional safety checks for invalid grades
    valid_train_mask = ~np.isnan(y_train) & (y_train >= 1) & (y_train <= 14)
    valid_test_mask = ~np.isnan(y_test) & (y_test >= 1) & (y_test <= 14)
    
    invalid_train = np.sum(~valid_train_mask)
    invalid_test = np.sum(~valid_test_mask)
    
    if invalid_train > 0:
        print(f"Warning: Filtering {invalid_train} training samples with invalid grades")
        print(f"  Invalid training grades: {y_train[~valid_train_mask][:10]}...")  # Show first 10
    
    if invalid_test > 0:
        print(f"Warning: Filtering {invalid_test} test samples with invalid grades")
        print(f"  Invalid test grades: {y_test[~valid_test_mask][:10]}...")  # Show first 10
    
    x_train = x_train[valid_train_mask]
    y_train = y_train[valid_train_mask]
    x_test = x_test[valid_test_mask]
    y_test = y_test[valid_test_mask]
    
    print(f"After filtering invalid grades:")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Training grade range: {y_train.min():.0f} - {y_train.max():.0f}")
    print(f"Test grade range: {y_test.min():.0f} - {y_test.max():.0f}")
    
    # Train model
    model = train_autokeras_model(x_train, y_train, x_test, y_test, 
                                 max_trials=max_trials, epochs=epochs, 
                                 model_dir=model_dir)
    
    # Evaluate model
    predictions, metrics = evaluate_model(model, x_test, y_test)
    
    # Save results
    save_model_and_results(model, predictions, metrics, y_test, output_dir)
    
    # Plot results
    if plot:
        plot_results(predictions, y_test, output_dir)
    
    return model, predictions, metrics


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Train MoonBoard climbing grade prediction model')
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--json-file', type=str,
                           help='Path to MoonBoard JSON data file')
    data_group.add_argument('--data-dir', type=str,
                           help='Directory with preprocessed data files')
    
    # Training parameters
    parser.add_argument('--max-trials', type=int, default=1,
                       help='Maximum AutoKeras trials (default: 1)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Training epochs (default: 3)')
    parser.add_argument('--test-size', type=int, default=1000,
                       help='Test set size when preparing from JSON (default: 1000)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output directories
    parser.add_argument('--output-dir', type=str, default='./model_output',
                       help='Directory to save results (default: ./model_output)')
    parser.add_argument('--model-dir', type=str, default='./moonboard_model',
                       help='Directory for AutoKeras model (default: ./moonboard_model)')
    
    # Options
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    try:
        model, predictions, metrics = train_moonboard_model(
            json_file=args.json_file,
            data_dir=args.data_dir,
            max_trials=args.max_trials,
            epochs=args.epochs,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            plot=not args.no_plot
        )
        
        print("\nModel training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Model saved to: {args.model_dir}")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())