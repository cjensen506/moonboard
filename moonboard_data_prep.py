#!/usr/bin/env python3
"""
MoonBoard Data Preparation Script
Handles data loading, preprocessing, and train-test split for MoonBoard climbing problems.
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import os


def letter_to_number(letter):
    """Convert letter to number (A=1, B=2, etc.)"""
    letter = letter.upper()
    if letter.isalpha() and len(letter) == 1:
        return ord(letter) - ord('A') + 1
    else:
        return None


def hold_to_coord(hold_string):
    """Convert hold string (e.g., 'E6') to coordinates [x, y]"""
    pattern = r'\d+'
    match = re.search(pattern, hold_string)
    x = int(match.group())
    
    pattern = r'[a-zA-Z]+'
    match = re.search(pattern, hold_string)
    y = match.group()
    y = letter_to_number(y)
    
    return [x, y]


def get_hold_list(moves):
    """Extract hold descriptions from moves list"""
    hold_list = []
    for move in moves:
        hold_list.append(move['description'])
    return hold_list


def problem_holds_to_array(all_holds):
    """Convert problem holds to 18x11 binary array"""
    problem_array = np.zeros((18, 11))
    for hold in all_holds:
        hold_coord = hold_to_coord(hold)
        problem_array[hold_coord[0] - 1, hold_coord[1] - 1] = 1
    return problem_array


def grade_to_numeric(grade):
    """Convert French climbing grade to numeric value"""
    grade_conversion = {
        '6B': 1, 
        '6B+': 2, 
        '6C': 3, 
        '6C+': 4,
        '7A': 5,
        '7A+': 6,
        '7B': 7,
        '7B+': 8,
        '7C': 9,
        '7C+': 10,
        '8A': 11,
        '8A+': 12,
        '8B': 13,
        '8B+': 14
    }
    if grade is None:
        return np.nan
    return grade_conversion.get(grade, np.nan)


def load_moonboard_data(json_file):
    """Load MoonBoard data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def preprocess_problems(data):
    """Process raw problem data into structured DataFrame"""
    problem_df = pd.DataFrame(columns=['name', 'grade', 'userGrade', 'apiId', 'moves'])
    
    for i, problem in tqdm(enumerate(data['data']), total=len(data['data']), desc="Processing problems"):
        name = problem['name']
        grade = problem['grade']
        userGrade = problem['userGrade']
        apiId = problem['apiId']
        hold_list = get_hold_list(problem['moves'])
        problem_df.loc[i] = [name, grade, userGrade, apiId, hold_list]
    
    # Convert grades to numeric
    problem_df['userGradeNumeric'] = problem_df['userGrade'].apply(grade_to_numeric)
    problem_df['gradeNumeric'] = problem_df['grade'].apply(grade_to_numeric)
    
    return problem_df


def build_training_arrays(problem_df):
    """Build training arrays from problem DataFrame"""
    # Filter out problems with missing grades
    valid_problems = problem_df.dropna(subset=['gradeNumeric'])
    
    print(f"Filtered {len(problem_df) - len(valid_problems)} problems with missing grades")
    print(f"Using {len(valid_problems)} problems with valid grades")
    
    num_of_problems = valid_problems.shape[0]
    x_all = np.zeros((num_of_problems, 18, 11))
    y_all = np.zeros(num_of_problems)
    
    for i, (index, problem) in tqdm(enumerate(valid_problems.iterrows()), total=num_of_problems, desc="Building training arrays"):
        problem_array = problem_holds_to_array(problem['moves'])
        x_all[i, :, :] = problem_array
        y_all[i] = problem['gradeNumeric']
    
    return x_all, y_all


def prepare_moonboard_data(json_file, test_size=1000, random_state=42, output_dir=None):
    """
    Complete data preparation pipeline for MoonBoard data
    
    Args:
        json_file (str): Path to JSON file containing MoonBoard problems
        test_size (int): Number of samples for test set
        random_state (int): Random seed for reproducibility
        output_dir (str): Directory to save processed data (optional)
    
    Returns:
        tuple: (x_train, x_test, y_train, y_test, problem_df)
    """
    
    print(f"Loading data from {json_file}...")
    data = load_moonboard_data(json_file)
    
    print("Preprocessing problems...")
    problem_df = preprocess_problems(data)
    
    print("Building training arrays...")
    x_all, y_all = build_training_arrays(problem_df)
    
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=random_state
    )
    
    print(f"Data split complete:")
    print(f"  Training set: {x_train.shape[0]} samples")
    print(f"  Test set: {x_test.shape[0]} samples")
    print(f"  Feature shape: {x_train.shape[1:]}")
    print(f"  Grade range: {y_all.min():.0f} - {y_all.max():.0f}")
    print(f"  Training grades range: {y_train.min():.0f} - {y_train.max():.0f}")
    print(f"  Test grades range: {y_test.min():.0f} - {y_test.max():.0f}")
    
    # Save processed data if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'x_train.npy'), x_train)
        np.save(os.path.join(output_dir, 'x_test.npy'), x_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        problem_df.to_pickle(os.path.join(output_dir, 'problem_df.pkl'))
        
        print(f"Processed data saved to {output_dir}")
    
    return x_train, x_test, y_train, y_test, problem_df


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Prepare MoonBoard data for machine learning')
    parser.add_argument('json_file', help='Path to MoonBoard JSON data file')
    parser.add_argument('--test-size', type=int, default=1000, 
                       help='Number of samples for test set (default: 1000)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save processed data files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} not found")
        return 1
    
    try:
        x_train, x_test, y_train, y_test, problem_df = prepare_moonboard_data(
            args.json_file,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir
        )
        
        print("\nData preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())