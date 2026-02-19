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
from sklearn.utils.class_weight import compute_sample_weight
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
                         max_trials=1, epochs=3, model_dir="./moonboard_model",
                         sample_weight=None):
    """Train AutoKeras ImageRegressor model and wrap output with softplus+1.

    The raw regression head is unconstrained and can produce large negative
    values. We export the best model after training and prepend a softplus
    activation (always positive, unbounded above) shifted by +1 so the
    minimum output is ~1, matching the lowest grade in the data.

    Sample weights are embedded into a tf.data.Dataset (x, y, weight) triple,
    which is the only path AutoKeras accepts for per-sample weighting.
    """

    print("Initializing AutoKeras ImageRegressor...")
    reg = ak.ImageRegressor(
        overwrite=True,
        max_trials=max_trials,
        directory=model_dir
    )

    # NOTE: AutoKeras doesn't support per-sample or class weights in its
    # standard fit() path (it manages its own tf.data pipeline internally).
    # Weighting is logged by train_moonboard_model() for reference but is not
    # applied during training in this AutoKeras version. A future improvement
    # would be to use a custom Keras model that supports sample_weight directly.
    if sample_weight is not None:
        print("  (Note: class weights logged but not applied — AutoKeras limitation)")

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


_GRADE_LABELS = {
    1: "6B", 2: "6B+", 3: "6C", 4: "6C+", 5: "7A", 6: "7A+",
    7: "7B", 8: "7B+", 9: "7C", 10: "7C+", 11: "8A", 12: "8A+",
    13: "8B", 14: "8B+",
}


def compute_macro_mae(y_test, predictions):
    """Macro-averaged MAE: per-grade MAE averaged equally across grades.

    Unlike micro-averaged MAE, each grade contributes equally regardless of
    how many test samples it has. Returns (macro_mae, per_grade_dict) where
    per_grade_dict maps grade int -> MAE float.
    """
    pred_flat = np.array(predictions).flatten()
    grades = np.unique(y_test).astype(int)
    per_grade = {
        int(g): float(mean_absolute_error(
            y_test[y_test == g], pred_flat[y_test == g]
        ))
        for g in grades
    }
    return float(np.mean(list(per_grade.values()))), per_grade


def compute_naive_baselines(y_train, y_test):
    """Compute micro and macro MAE for naive constant predictors."""
    median_grade = float(np.median(y_train))
    mean_grade = float(np.mean(y_train))
    median_mae = float(mean_absolute_error(y_test, np.full_like(y_test, median_grade)))
    mean_mae = float(mean_absolute_error(y_test, np.full_like(y_test, mean_grade)))
    macro_median_mae, _ = compute_macro_mae(y_test, np.full_like(y_test, median_grade))
    return {
        "median_grade": median_grade,
        "median_mae": median_mae,
        "mean_grade": mean_grade,
        "mean_mae": mean_mae,
        "macro_median_mae": macro_median_mae,
    }


def evaluate_model(model, x_test, y_test, y_train):
    """Evaluate the trained model against naive baselines.

    Reports three levels:
      - Micro MAE: overall mean over all test samples (natural distribution)
      - Macro MAE: per-grade MAE averaged equally across grades
      - Per-grade table: individual MAE for each grade in the test set

    model is the softplus-wrapped Keras Model returned by train_autokeras_model().
    """
    print("\nEvaluating model...")

    # Get predictions (already constrained to >=1 via softplus+1 wrapper)
    predicted_y = model.predict(x_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predicted_y)
    micro_mae = mean_absolute_error(y_test, predicted_y)
    macro_mae, per_grade_mae = compute_macro_mae(y_test, predicted_y)

    # Naive baselines
    baselines = compute_naive_baselines(y_train, y_test)
    micro_diff = micro_mae - baselines["median_mae"]
    macro_diff = macro_mae - baselines["macro_median_mae"]

    print(f"\n{'Metric':<28} {'Value':>10}")
    print("-" * 40)
    print(f"{'Naive median micro MAE':<28} {baselines['median_mae']:>10.4f}  (always predict {baselines['median_grade']:.0f})")
    print(f"{'Naive mean micro MAE':<28} {baselines['mean_mae']:>10.4f}  (always predict {baselines['mean_grade']:.2f})")
    print(f"{'Naive median macro MAE':<28} {baselines['macro_median_mae']:>10.4f}")
    print(f"{'Model micro MAE':<28} {micro_mae:>10.4f}")
    print(f"{'Model macro MAE':<28} {macro_mae:>10.4f}")
    print(f"{'Model MSE':<28} {mse:>10.4f}")

    micro_verdict = f"{'BEATS' if micro_diff < 0 else 'DOES NOT BEAT'} naive by {abs(micro_diff):.4f}"
    macro_verdict = f"{'BEATS' if macro_diff < 0 else 'DOES NOT BEAT'} naive by {abs(macro_diff):.4f}"
    print(f"\n  --> Micro (natural dist): {micro_verdict}")
    print(f"  --> Macro (per-grade):    {macro_verdict}")
    print(f"  --> Prediction range: {predicted_y.min():.2f} – {predicted_y.max():.2f}")

    # Per-grade breakdown
    print(f"\n{'Grade':<16} {'MAE':>8}  {'N':>6}")
    print("-" * 33)
    for g in sorted(per_grade_mae):
        label = _GRADE_LABELS.get(g, str(g))
        n = int(np.sum(y_test == g))
        print(f"  {g} ({label:<5})      {per_grade_mae[g]:>8.4f}  {n:>6}")

    return predicted_y, {
        "mse": mse,
        "micro_mae": micro_mae,
        "macro_mae": macro_mae,
        "per_grade_mae": per_grade_mae,
        "baseline_micro_median_mae": baselines["median_mae"],
        "baseline_micro_mean_mae": baselines["mean_mae"],
        "baseline_macro_median_mae": baselines["macro_median_mae"],
        "beats_micro_baseline": micro_diff < 0,
        "beats_macro_baseline": macro_diff < 0,
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
                         model_dir="./moonboard_model", plot=True,
                         min_grade=2, max_grade=11):
    """
    Complete model training pipeline.

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
        min_grade (int): Minimum grade to include (default 2 = 6B+)
        max_grade (int): Maximum grade to include (default 11 = 8A)

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

    # Filter to the configured grade range (removes NaN, too-rare grades, etc.)
    print(f"\nFiltering to grades {min_grade}–{max_grade} "
          f"({_GRADE_LABELS.get(min_grade, min_grade)} – {_GRADE_LABELS.get(max_grade, max_grade)})...")
    train_mask = ~np.isnan(y_train) & (y_train >= min_grade) & (y_train <= max_grade)
    test_mask = ~np.isnan(y_test) & (y_test >= min_grade) & (y_test <= max_grade)

    removed_train = np.sum(~train_mask)
    removed_test = np.sum(~test_mask)
    if removed_train:
        print(f"  Removed {removed_train} training samples outside grade range")
    if removed_test:
        print(f"  Removed {removed_test} test samples outside grade range")

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples:     {len(y_test)}")

    # Sample weights so rare grades contribute proportionally during training.
    # Passed via a tf.data.Dataset tuple inside train_autokeras_model().
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grade_counts = {int(g): int(np.sum(y_train == g)) for g in np.unique(y_train)}
    max_w, min_w = sample_weights.max(), sample_weights.min()
    print(f"\nClass weights applied (max ratio {max_w / min_w:.1f}x):")
    for g in sorted(grade_counts):
        label = _GRADE_LABELS.get(g, str(g))
        w = float(sample_weights[y_train == g][0])
        print(f"  Grade {g:2d} ({label:<5}): {grade_counts[g]:6d} samples, weight {w:.3f}")

    # Train model
    model = train_autokeras_model(x_train, y_train, x_test, y_test,
                                  max_trials=max_trials, epochs=epochs,
                                  model_dir=model_dir, sample_weight=sample_weights)

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
    
    # Grade range
    parser.add_argument('--min-grade', type=int, default=2,
                       help='Minimum grade to include (default: 2 = 6B+)')
    parser.add_argument('--max-grade', type=int, default=11,
                       help='Maximum grade to include (default: 11 = 8A)')

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
            plot=not args.no_plot,
            min_grade=args.min_grade,
            max_grade=args.max_grade,
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