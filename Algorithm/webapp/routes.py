"""
SmartSelect Platform - API Routes Module
RESTful API endpoints and request handlers for web services
Developed by: Fadi Younes (API Development Engineer)
"""

from flask import Blueprint, render_template, request, jsonify, send_file
import os
import sys
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Add parent directory to path to import genetic algorithm modules
# This ensures all modules are accessible
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.evolution.ga_engine import GeneticAlgorithm
from core.data.loader import DataLoader
from core.data.preprocessor import DataPreprocessor
from core.analysis.traditional_methods import TraditionalFeatureSelection
from core.models.evaluator import ModelEvaluator

# Create Blueprint
main_bp = Blueprint('main', __name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@main_bp.route('/')
def index():
    """
    Home page route
    Displays welcome page and navigation
    """
    return render_template('index.html')


@main_bp.route('/upload')
def upload_page():
    """
    Upload page route
    Displays file upload interface
    """
    return render_template('upload.html')


@main_bp.route('/experiment')
def experiment_page():
    """
    Experiment lab page route
    Interactive experiment interface with real-time monitoring
    """
    return render_template('experiment.html')


@main_bp.route('/results-dashboard')
def results_dashboard():
    """
    Advanced results dashboard route
    Displays comprehensive analytics and visualizations
    """
    return render_template('results_dashboard.html')


@main_bp.route('/team')
def team_page():
    """
    Team page route
    Displays team members and their responsibilities
    """
    return render_template('team.html')


@main_bp.route('/api/upload', methods=['POST'])
def upload_file():
    """
    API endpoint for file upload
    Accepts CSV/Excel files and validates them
    """
    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed.'}), 400
    
    try:
        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        
        # Quick validation: try to read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Get basic file info
        file_info = {
            'filename': filename,
            'filepath': filepath,
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist(),
            'success': True
        }
        
        return jsonify(file_info), 200
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


@main_bp.route('/api/run-ga', methods=['POST'])
def run_genetic_algorithm():
    """
    API endpoint to run genetic algorithm
    Receives parameters and executes GA for feature selection
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        filepath = data.get('filepath')
        population_size = data.get('population_size', 50)
        generations = data.get('generations', 100)
        crossover_rate = data.get('crossover_rate', 0.8)
        mutation_rate = data.get('mutation_rate', 0.1)
        model_type = data.get('model_type', 'random_forest')
        target_column = data.get('target_column', 'target')
        
        # Validate filepath
        if not filepath:
            return jsonify({'error': 'Filepath is required'}), 400
        
        # Convert relative paths to absolute paths (for uploads and datasets)
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            filepath = os.path.join(base_dir, filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404
        
        # Load data
        loader = DataLoader()
        df = loader.load_file(filepath)
        
        # Check if target column exists
        if target_column not in df.columns:
            return jsonify({
                'error': f'Target column "{target_column}" not found in dataset. Available columns: {df.columns.tolist()}'
            }), 400
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        
        # Handle missing values and encode categorical
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        
        # Separate features and target
        X, y = preprocessor.separate_features_target(df_encoded, target_column)
        
        # Normalize features
        X_normalized = preprocessor.normalize_features(X.values, method='standard')
        
        # Store feature names for later reference
        feature_names = X.columns.tolist()
        
        # Create and run Genetic Algorithm
        ga = GeneticAlgorithm(
            population_size=population_size,
            n_generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_rate=0.1,
            alpha=0.9,
            model_type=model_type,
            verbose=False  # Set to False for web API
        )
        
        # Run GA
        best_chromosome = ga.fit(X_normalized, y.values)
        
        # Get results
        selected_indices = ga.get_selected_features().tolist()
        selected_feature_names = [feature_names[i] for i in selected_indices]
        history = ga.get_history()
        
        # Prepare result
        result = {
            'success': True,
            'selected_features': selected_feature_names,
            'selected_indices': selected_indices,
            'fitness_score': float(best_chromosome.fitness),
            'accuracy': float(best_chromosome.accuracy),
            'n_selected_features': int(best_chromosome.n_selected_features),
            'n_total_features': len(feature_names),
            'feature_reduction_percent': float((1 - best_chromosome.n_selected_features / len(feature_names)) * 100),
            'generation_history': {
                'best_fitness': [float(x) for x in history['best_fitness']],
                'avg_fitness': [float(x) for x in history['avg_fitness']],
                'worst_fitness': [float(x) for x in history['worst_fitness']],
                'best_accuracy': [float(x) for x in history['best_accuracy']],
                'avg_features': [float(x) for x in history['avg_features']]
            },
            'message': 'Genetic algorithm completed successfully'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_genetic_algorithm: {error_details}")
        return jsonify({'error': f'Error running GA: {str(e)}'}), 500


@main_bp.route('/api/comparison', methods=['POST'])
def run_comparison():
    """
    API endpoint to run comparison analysis
    Compares GA results with traditional methods
    """
    try:
        data = request.get_json()
        
        # Extract parameters
        filepath = data.get('filepath')
        target_column = data.get('target_column', 'target')
        k_features = data.get('k_features', 10)
        
        # Validate filepath
        if not filepath:
            return jsonify({'error': 'Filepath is required'}), 400
        
        # Convert relative paths to absolute paths
        if not os.path.isabs(filepath):
            base_dir = os.path.dirname(os.path.dirname(__file__))
            filepath = os.path.join(base_dir, filepath)
        
        # Check if file exists
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404
        
        # Load and preprocess data
        loader = DataLoader()
        df = loader.load_file(filepath)
        
        # Check if target column exists
        if target_column not in df.columns:
            return jsonify({
                'error': f'Target column "{target_column}" not found. Available: {df.columns.tolist()}'
            }), 400
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        df_encoded = preprocessor.encode_categorical_features(df_clean)
        X, y = preprocessor.separate_features_target(df_encoded, target_column)
        X_normalized = preprocessor.normalize_features(X.values, method='standard')
        feature_names = X.columns.tolist()
        
        # Initialize traditional feature selection
        selector = TraditionalFeatureSelection()
        evaluator = ModelEvaluator()
        
        # Run traditional methods
        methods_results = {}
        
        # Chi-Square
        try:
            chi_indices, chi_scores = selector.chi_square_selection(X_normalized, y.values, k=k_features)
            chi_features = [feature_names[i] for i in chi_indices]
            chi_metrics = evaluator.evaluate_features(X_normalized[:, chi_indices], y.values)
            methods_results['chi_square'] = {
                'features': chi_features,
                'n_features': len(chi_features),
                'accuracy': float(chi_metrics['accuracy']),
                'precision': float(chi_metrics['precision']),
                'recall': float(chi_metrics['recall']),
                'f1_score': float(chi_metrics['f1_score']),
                'time': float(chi_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Chi-Square error: {str(e)}")
            methods_results['chi_square'] = {'error': str(e)}
        
        # ANOVA F-test
        try:
            anova_indices, anova_scores = selector.anova_f_test(X_normalized, y.values, k=k_features)
            anova_features = [feature_names[i] for i in anova_indices]
            anova_metrics = evaluator.evaluate_features(X_normalized[:, anova_indices], y.values)
            methods_results['anova_f'] = {
                'features': anova_features,
                'n_features': len(anova_features),
                'accuracy': float(anova_metrics['accuracy']),
                'precision': float(anova_metrics['precision']),
                'recall': float(anova_metrics['recall']),
                'f1_score': float(anova_metrics['f1_score']),
                'time': float(anova_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"ANOVA error: {str(e)}")
            methods_results['anova_f'] = {'error': str(e)}
        
        # Mutual Information
        try:
            mi_indices, mi_scores = selector.mutual_information(X_normalized, y.values, k=k_features)
            mi_features = [feature_names[i] for i in mi_indices]
            mi_metrics = evaluator.evaluate_features(X_normalized[:, mi_indices], y.values)
            methods_results['mutual_info'] = {
                'features': mi_features,
                'n_features': len(mi_features),
                'accuracy': float(mi_metrics['accuracy']),
                'precision': float(mi_metrics['precision']),
                'recall': float(mi_metrics['recall']),
                'f1_score': float(mi_metrics['f1_score']),
                'time': float(mi_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Mutual Information error: {str(e)}")
            methods_results['mutual_info'] = {'error': str(e)}
        
        # Random Forest Importance
        try:
            rf_indices, rf_scores = selector.random_forest_importance(X_normalized, y.values)
            # Select top k features
            if len(rf_indices) > k_features:
                top_k_idx = np.argsort(rf_scores[rf_indices])[-k_features:]
                rf_indices = rf_indices[top_k_idx]
            rf_features = [feature_names[i] for i in rf_indices]
            rf_metrics = evaluator.evaluate_features(X_normalized[:, rf_indices], y.values)
            methods_results['rf_importance'] = {
                'features': rf_features,
                'n_features': len(rf_features),
                'accuracy': float(rf_metrics['accuracy']),
                'precision': float(rf_metrics['precision']),
                'recall': float(rf_metrics['recall']),
                'f1_score': float(rf_metrics['f1_score']),
                'time': float(rf_metrics.get('time', 0))
            }
        except Exception as e:
            print(f"Random Forest error: {str(e)}")
            methods_results['rf_importance'] = {'error': str(e)}
        
        # Statistical analysis
        statistical_analysis = {
            'total_features': len(feature_names),
            'methods_compared': len([m for m in methods_results.values() if 'error' not in m]),
            'summary': 'Comparison completed successfully'
        }
        
        result = {
            'success': True,
            'traditional_results': methods_results,
            'statistical_analysis': statistical_analysis,
            'message': 'Comparison analysis completed'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_comparison: {error_details}")
        return jsonify({'error': f'Error running comparison: {str(e)}'}), 500


@main_bp.route('/api/datasets')
def list_datasets():
    """
    API endpoint to list available sample datasets
    Returns list of pre-loaded datasets for testing
    """
    try:
        datasets_dir = 'datasets'
        if not os.path.exists(datasets_dir):
            return jsonify({'datasets': []}), 200
        
        datasets = []
        for filename in os.listdir(datasets_dir):
            if filename.endswith(('.csv', '.xlsx', '.xls')):
                filepath = os.path.join(datasets_dir, filename)
                datasets.append({
                    'name': filename,
                    'path': filepath
                })
        
        return jsonify({'datasets': datasets}), 200
        
    except Exception as e:
        return jsonify({'error': f'Error listing datasets: {str(e)}'}), 500

