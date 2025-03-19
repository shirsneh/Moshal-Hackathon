import numpy as np
from Preprocessing import HSI_processing
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import argparse
import os
import sys

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_input_file(file_path):
    """
    Validate the input HDR file
    
    Args:
        file_path (str): Path to the input file
        
    Raises:
        ValidationError: If the file is invalid
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"Input file not found: {file_path}")
    
    if not file_path.lower().endswith('.hdr'):
        raise ValidationError(f"Input file must be an HDR file: {file_path}")
    
    # Check for the corresponding IMG data file
    base_path = os.path.splitext(file_path)[0]
    img_file = base_path + '.IMG'
    if not os.path.exists(img_file):
        raise ValidationError(f"Associated file not found: {img_file}")

def validate_delta(delta):
    """
    Validate the delta threshold value
    
    Args:
        delta (float): The threshold value
        
    Raises:
        ValidationError: If delta is invalid
    """
    if not (0 < delta < 1):
        raise ValidationError(f"Delta must be between 0 and 1, got: {delta}")

def validate_output_dir(output_dir):
    """
    Validate the output directory
    
    Args:
        output_dir (str): Path to the output directory
        
    Raises:
        ValidationError: If the directory is invalid
    """
    if output_dir:
        # Check if directory exists or can be created
        try:
            os.makedirs(output_dir, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValidationError(f"Cannot create output directory {output_dir}: {str(e)}")
        
        # Check if directory is writable
        if not os.access(output_dir, os.W_OK):
            raise ValidationError(f"Output directory is not writable: {output_dir}")

class MineralClassifier:
    def __init__(self, data_path, delta=0.3, output_dir=None):
        """
        Initialize the MineralClassifier
        
        Args:
            data_path (str): Path to the hyperspectral image data
            delta (float): Threshold for angle-based classification
            output_dir (str): Directory to save output files
            
        Raises:
            ValidationError: If any parameters are invalid
        """
        # Validate input parameters
        validate_input_file(data_path)
        validate_delta(delta)
        validate_output_dir(output_dir)
        
        self.data_path = data_path
        self.delta = delta
        self.output_dir = output_dir
        self.processor = HSI_processing()
        
    def process_image(self):
        """Process the hyperspectral image and perform mineral classification"""
        # Load and preprocess data
        self.processor.load_data(self.data_path)
        self.processor.load_materials()
        
        # Perform clustering
        self.processor.KMeans(self.data_path)
        self.processor.get_cluster_mean()
        
        # Classify minerals
        self.processor.get_angle_between_mean_vectors(self.delta)
        
        return self.generate_mineral_map()
    
    def generate_mineral_map(self):
        """Generate a colored map of mineral classifications"""
        # Create a color map for different minerals
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        mineral_names = list(self.processor.materials_dict.keys())
        
        # Create the mineral map
        mineral_map = np.zeros(self.processor.cluster_mtxs[0].shape + (3,))
        
        for i in range(len(self.processor.classification_mtx)):
            for mineral_idx in range(len(mineral_names)):
                mask = self.processor.cluster_mtxs[i] == mineral_idx
                if mineral_idx < len(colors):
                    color = np.array(plt.cm.colors.to_rgb(colors[mineral_idx]))
                    mineral_map[mask] = color
                    
        return mineral_map, mineral_names
    
    def visualize_results(self, mineral_map, mineral_names, save=False):
        """
        Visualize the mineral classification results
        
        Args:
            mineral_map: The generated mineral distribution map
            mineral_names: List of mineral names
            save (bool): Whether to save the plot to a file
        """
        plt.figure(figsize=(12, 8))
        plt.imshow(mineral_map)
        plt.title("Mineral Distribution Map")
        
        # Create legend
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c) 
                         for c in plt.cm.colors.to_rgb(colors[:len(mineral_names)])]
        plt.legend(legend_elements, mineral_names, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        
        if save and self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, 'mineral_map.png')
            plt.savefig(output_path)
            print(f"Saved mineral map to: {output_path}")
        else:
            plt.show()

def parse_arguments():
    """
    Parse and validate command line arguments
    
    Returns:
        argparse.Namespace: Validated command line arguments
        
    Raises:
        ValidationError: If any arguments are invalid
    """
    parser = argparse.ArgumentParser(
        description='MoonMineralMapper - Analyze lunar images for mineral distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python model.py --input data/lunar_image.HDR --delta 0.3 --output results/
  python model.py --input data/lunar_image.HDR --show-plot
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to the hyperspectral image data (HDR file)'
    )
    
    parser.add_argument(
        '--delta', '-d',
        type=float,
        default=0.3,
        help='Threshold for angle-based classification (default: 0.3)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Directory to save output files (if not specified, will show plot)'
    )
    
    parser.add_argument(
        '--show-plot',
        action='store_true',
        help='Display the plot instead of saving to file'
    )
    
    args = parser.parse_args()
    
    # Validate all arguments
    try:
        validate_input_file(args.input)
        validate_delta(args.delta)
        validate_output_dir(args.output)
    except ValidationError as e:
        parser.error(str(e))
    
    return args

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create classifier instance
        classifier = MineralClassifier(
            data_path=args.input,
            delta=args.delta,
            output_dir=args.output
        )
        
        # Process the image
        print("Processing image...")
        mineral_map, mineral_names = classifier.process_image()
        
        # Visualize or save results
        if args.show_plot or not args.output:
            print("Displaying mineral distribution map...")
            classifier.visualize_results(mineral_map, mineral_names, save=False)
        else:
            print("Saving mineral distribution map...")
            classifier.visualize_results(mineral_map, mineral_names, save=True)
            
    except ValidationError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)
