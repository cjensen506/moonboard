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
    """Train AutoKeras ImageRegressor model and wrap output with softplus+1.

    The raw regression head is unconstrained and can produce large negative
    values. We export the best model after training and prepend a softplus
    activation (always positive, unbounded above) shifted by +1 so the
    minimum output is ~1, matching the lowest grade in the data.
    """

    print("Initializing AutoKeras ImageRegressor...")
    reg = ak.ImageRegressor(
        overwrite=True,
        max_trials=max_trials,
        directory=model_dir
    )

    print(f"Training model with {max_trials} trial(s) and {epochs} epochs...")
    reg.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    # Wrap the best model with a softplus+1 output so predictions are
    # always >= ~1 without imposing an upper bound on the grade scale.
    print("Wrapping best model output with softplus+1...")
    best_model = reg.export_model()
    constrained_output = tf.keras.layers.Lambda(
        lambda t: tf.nn.softplus(t) + 1,
        name="softplus_grade_output"
    )(best_model.output)
    constrained_model = tf.keras.Model(
        inputs=best_model.input, outputs=constrained_output
    )

    return constrained_model


def compute_naive_baselines(y_train, y_test):
    """Compute MAE for naive constant predictors using training set statistics."""
    median_grade = float(np.median(y_train))
    mean_grade = float(np.mean(y_train))
    median_mae = float(mean_absolute_error(y_test, np.full_like(y_test, median_grade)))
    mean_mae = float(mean_absolute_error(y_test, np.full_like(y_test, mean_grade)))
    return {
        "median_grade": median_grade,
        "median_mae": median_mae,
        "mean_grade": mean_grade,
        "mean_mae": mean_mae,
    }


def evaluate_model(model, x_test, y_test, y_train):
    """Evaluate the trained model against naive baselines.

    model is the softplus-wrapped Keras Model returned by train_autokeras_model().
    """
    print("\nEvaluating model...")

    # Get predictions (already constrained to >=1 via softplus+1 wrapper)
    predicted_y = model.predict(x_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predicted_y)
    mae = mean_absolute_error(y_test, predicted_y)

    # Naive baseline comparison
    baselines = compute_naive_baselines(y_train, y_test)
    diff = mae - baselines["median_mae"]
    beats = diff < 0

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Naive median MAE':<25} {baselines['median_mae']:>10.4f}  (always predict {baselines['median_grade']:.0f})")
    print(f"{'Naive mean MAE':<25} {baselines['mean_mae']:>10.4f}  (always predict {baselines['mean_grade']:.2f})")
    print(f"{'Model MAE':<25} {mae:>10.4f}")
    print(f"{'Model MSE':<25} {mse:>10.4f}")
    verdict = f"{'BEATS' if beats else 'DOES NOT BEAT'} naive baseline by {abs(diff):.4f}"
    print(f"\n  --> {verdict}")
    print(f"  --> Prediction range: {predicted_y.min():.2f} – {predicted_y.max():.2f}")

    return predicted_y, {
        "mse": mse,
        "mae": mae,
        "baseline_median_mae": baselines["median_mae"],
        "baseline_mean_mae": baselines["mean_mae"],
        "beats_baseline": beats,
    }


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


def plot_results(predictions, y_test, y_train, output_dir=None):
    """Plot prediction vs actual results with naive baseline annotations."""
    pred_flat = predictions.flatten()
    baselines = compute_naive_baselines(y_train, y_test)
    naive_median = baselines["median_grade"]
    naive_mae = baselines["median_mae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot
    axes[0].scatter(y_test, pred_flat, alpha=0.4)
    lims = [min(y_test.min(), pred_flat.min()), max(y_test.max(), pred_flat.max())]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel('Actual Grade')
    axes[0].set_ylabel('Predicted Grade')
    axes[0].set_title('Predicted vs Actual Grades')
    axes[0].grid(True, alpha=0.3)

    # Residuals plot — shaded band shows naive baseline error range
    residuals = pred_flat - y_test
    axes[1].scatter(y_test, residuals, alpha=0.4)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].axhspan(-naive_mae, naive_mae, alpha=0.12, color='green',
                    label=f'Naive baseline ±{naive_mae:.2f}')
    axes[1].set_xlabel('Actual Grade')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Grade distribution comparison with naive baseline marker
    bins = np.arange(0.5, max(y_test.max(), pred_flat.max()) + 1.5, 1)
    axes[2].hist(y_test, bins=bins, alpha=0.6, label='Actual', color='steelblue')
    axes[2].hist(pred_flat, bins=bins, alpha=0.6, label='Predicted', color='orange')
    axes[2].axvline(x=naive_median, color='green', linestyle='--', lw=1.5,
                    label=f'Naive baseline ({naive_median:.0f})')
    axes[2].set_xlabel('Grade')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Grade Distribution')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

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
    predictions, metrics = evaluate_model(model, x_test, y_test, y_train)

    # Save results
    save_model_and_results(model, predictions, metrics, y_test, output_dir)

    # Plot results
    if plot:
        plot_results(predictions, y_test, y_train, output_dir)
    
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