import imagehash
from PIL import Image

class ImageHasher:
    def __init__(self):
        pass
        
    def compute_hashes(self, image_path):
        """
        Computes Average, Perceptual, and Difference hashes for an image.
        """
        try:
            img = Image.open(image_path)
            hashes = {
                'aHash': str(imagehash.average_hash(img)),
                'pHash': str(imagehash.phash(img)),
                'dHash': str(imagehash.dhash(img))
            }
            return hashes
        except Exception as e:
            return {'error': str(e)}

    def compare_hashes(self, hash1, hash2):
        """
        Compares two hashes (must be of the same type).
        Returns the Hamming distance. Lower distance means higher similarity.
        Typically, distance <= 5 means they are likely the same image.
        """
        try:
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2
        except Exception as e:
            return -1

    def assess_integrity(self, target_hashes, reference_hashes=None):
        """
        If reference_hashes are provided, compares them.
        Otherwise, it just returns a generic score (usually 100 if we can't compare).
        """
        if not reference_hashes:
            return {'score': 100, 'message': 'No reference provided for hash comparison. Integrity assumed.'}
            
        distances = {}
        total_dist = 0
        
        for key in ['aHash', 'pHash', 'dHash']:
            if key in target_hashes and key in reference_hashes:
                dist = self.compare_hashes(target_hashes[key], reference_hashes[key])
                distances[key] = dist
                total_dist += dist
                
        # Simple scoring mechanism based on total distance
        # Max distance for 64-bit hash is 64. Total for 3 hashes is 192.
        # If total_dist is 0, score is 100.
        score = max(0, 100 - (total_dist * 2))
        
        message = "Hashes match perfectly." if total_dist == 0 else "Hashes differ, possible manipulation."
        if score < 40:
            message = "Significant hash differences, high probability of manipulation."
            
        return {
            'score': score,
            'message': message,
            'distances': distances
        }
