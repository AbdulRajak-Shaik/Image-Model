class TrustEngine:
    def __init__(self, deepfake_weight=0.4, metadata_weight=0.2, ela_weight=0.2, hash_weight=0.2):
        self.weights = {
            'deepfake': deepfake_weight,
            'metadata': metadata_weight,
            'ela': ela_weight,
            'hash': hash_weight
        }

    def calculate_score(self, df_conf, df_pred, meta_score, ela_avg_error, ela_max_error, hash_score):
        """
        Calculates the final Trust Score between 0 and 100.
        
        Args:
            df_conf (float): Deepfake confidence percentage (0-100).
            df_pred (str): 'Real' or 'Fake'.
            meta_score (float): Metadata integrity score (0-100).
            ela_avg_error (float): ELA average error.
            ela_max_error (float): ELA max error.
            hash_score (float): Hash integrity score (0-100).
        """
        
        # Deepfake Score contribution
        # If model says Real with 90% confidence, score is 90
        # If model says Fake with 90% confidence, score is 10 (100 - 90)
        df_base = df_conf if df_pred == 'Real' else (100 - df_conf)
        df_contribution = df_base * self.weights['deepfake']
        
        # Metadata Score contribution
        meta_contribution = meta_score * self.weights['metadata']
        
        # ELA Score contribution
        # We need to map ELA errors to a 0-100 score. 
        # Typically, avg_error < 5 is good. > 15 is very bad.
        ela_score_base = 100 - (ela_avg_error * 5)
        # Penalize if max error is very high
        if ela_max_error > 80:
            ela_score_base -= 20
        ela_score = max(0, min(100, ela_score_base))
        ela_contribution = ela_score * self.weights['ela']
        
        # Hash Score contribution
        hash_contribution = hash_score * self.weights['hash']
        
        total_trust_score = df_contribution + meta_contribution + ela_contribution + hash_contribution
        
        # Determine interpretation
        interpretation = self.get_interpretation(total_trust_score)
        
        return {
            'trust_score': round(total_trust_score, 2),
            'interpretation': interpretation,
            'breakdown': {
                'deepfake_score': round(df_base, 2),
                'metadata_score': round(meta_score, 2),
                'ela_score': round(ela_score, 2),
                'hash_score': round(hash_score, 2)
            }
        }

    def get_interpretation(self, score):
        if score >= 90:
            return "Highly Authentic"
        elif score >= 70:
            return "Likely Authentic"
        elif score >= 40:
            return "Suspicious"
        else:
            return "Highly Manipulated"
