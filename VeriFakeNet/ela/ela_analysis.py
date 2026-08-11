import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

class ELAAnalyzer:
    def __init__(self, quality=90, multiplier=15):
        self.quality = quality
        self.multiplier = multiplier

    def perform_ela(self, image_path):
        """
        Performs Error Level Analysis on the given image.
        Returns the ELA image, average error, and max error.
        """
        # Save a temporary JPEG with the given quality
        temp_filename = 'temp_ela.jpg'
        
        # Load the original image
        original = Image.open(image_path).convert('RGB')
        
        # Save it to trigger compression
        original.save(temp_filename, 'JPEG', quality=self.quality)
        
        # Load the compressed image
        compressed = Image.open(temp_filename)
        
        # Calculate the absolute difference
        ela_image = ImageChops.difference(original, compressed)
        
        # Get the extrema (min and max) for each channel to scale
        extrema = ela_image.getextrema()
        
        # Get the maximum difference across all channels
        max_diff = max([ex[1] for ex in extrema])
        
        if max_diff == 0:
            max_diff = 1 # Avoid division by zero
            
        # Scale to improve visibility
        scale = 255.0 / max_diff
        
        # Enhance the difference image
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        # Calculate statistics based on the difference array
        diff_array = np.array(ImageChops.difference(original, compressed))
        avg_error = np.mean(diff_array)
        max_error = np.max(diff_array)
        
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return ela_image, float(avg_error), float(max_error)

    def is_manipulated(self, avg_error, max_error, threshold_avg=5.0, threshold_max=50.0):
        """
        Heuristic to determine if the image might be manipulated based on ELA.
        Returns a boolean.
        """
        return avg_error > threshold_avg or max_error > threshold_max
