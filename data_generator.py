"""
Synthetic ECG Sequence Generator for AV Block Patterns
Based on clinical guidelines and expert consultation

Authors: Salma Bellaou, Asma Khamri
Data validated by: Dr. Mohammed Alami (Cardiology, CHU Rabat)
"""

import random
from typing import List, Tuple
from enum import Enum
import json


class AVBlockType(Enum):
    """AV block types for synthetic data generation"""
    FIRST_DEGREE = "first_degree"
    MOBITZ_I = "mobitz_i"
    MOBITZ_II = "mobitz_ii"
    THIRD_DEGREE = "third_degree"
    NORMAL = "normal"


class SyntheticECGGenerator:
    """
    Generate synthetic ECG symbolic sequences based on clinical patterns.
    
    Reference: Clinical guidelines validated with Dr. Mohammed Alami,
    Cardiologist at CHU Rabat, Morocco (consultation: December 2024)
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility"""
        random.seed(seed)
    
    def generate_first_degree(self, num_beats: int = 5) -> Tuple[List[str], dict]:
        """
        Generate First-Degree AV Block pattern.
        
        Clinical characteristics:
        - Constant PR prolongation (>200ms)
        - 1:1 AV conduction (all P-waves conduct)
        - PR interval remains stable across beats
        
        Args:
            num_beats: Number of cardiac cycles to generate
        
        Returns:
            Tuple of (symbolic sequence, metadata)
        """
        sequence = []
        pr_duration = random.randint(210, 280)  # Consistently prolonged
        
        for i in range(num_beats):
            sequence.extend(["P", "PRlong", "R"])
        
        metadata = {
            "type": AVBlockType.FIRST_DEGREE.value,
            "pr_duration_ms": pr_duration,
            "num_beats": num_beats,
            "description": "Constant PR prolongation with 1:1 conduction"
        }
        
        return sequence, metadata
    
    def generate_mobitz_i(self, cycles: int = 2) -> Tuple[List[str], dict]:
        """
        Generate Mobitz Type I (Wenckebach) pattern.
        
        Clinical characteristics:
        - Progressive PR prolongation
        - Culminates in dropped QRS (blocked beat)
        - PR intervals: 160ms → 180ms → 200ms → 230ms → DROP
        - Cycle then repeats
        
        Args:
            cycles: Number of Wenckebach cycles to generate
        
        Returns:
            Tuple of (symbolic sequence, metadata)
        """
        sequence = []
        
        for cycle in range(cycles):
            # Progressive prolongation (typically 3-4 beats before drop)
            progression_length = random.randint(3, 5)
            
            for i in range(progression_length):
                sequence.append("P")
                if i == 0:
                    sequence.append("PRincrease")  # First increase
                else:
                    sequence.append("PRincrease")  # Progressive increase
                sequence.append("R")
            
            # Final P-wave without QRS (dropped beat)
            sequence.extend(["P", "P"])  # Two consecutive P-waves
        
        metadata = {
            "type": AVBlockType.MOBITZ_I.value,
            "cycles": cycles,
            "progression_length": progression_length,
            "description": "Progressive PR prolongation with periodic dropped beats"
        }
        
        return sequence, metadata
    
    def generate_mobitz_ii(self, num_conducted: int = 3, num_blocks: int = 2) -> Tuple[List[str], dict]:
        """
        Generate Mobitz Type II pattern.
        
        Clinical characteristics:
        - Constant PR interval
        - Sudden dropped QRS without warning
        - More dangerous than Mobitz I
        - Often progresses to complete heart block
        
        Args:
            num_conducted: Number of normally conducted beats before block
            num_blocks: Number of blocked beats to include
        
        Returns:
            Tuple of (symbolic sequence, metadata)
        """
        sequence = []
        pr_duration = random.randint(160, 200)  # Normal or mildly prolonged
        
        for block in range(num_blocks):
            # Normal conducted beats with constant PR
            for i in range(num_conducted):
                sequence.extend(["P", "PRlong", "R"])
            
            # Sudden dropped beat (P-P pattern)
            sequence.extend(["P", "PRlong", "P"])
        
        metadata = {
            "type": AVBlockType.MOBITZ_II.value,
            "pr_duration_ms": pr_duration,
            "conducted_beats": num_conducted,
            "num_blocks": num_blocks,
            "description": "Constant PR with sudden dropped QRS complexes"
        }
        
        return sequence, metadata
    
    def generate_third_degree(self, num_beats: int = 6) -> Tuple[List[str], dict]:
        """
        Generate Third-Degree (Complete Heart Block) pattern.
        
        Clinical characteristics:
        - Complete AV dissociation
        - P-waves and QRS complexes occur independently
        - Variable PR intervals (no consistent relationship)
        - May see consecutive P-waves or R-waves
        
        Args:
            num_beats: Number of beats to generate
        
        Returns:
            Tuple of (symbolic sequence, metadata)
        """
        sequence = []
        
        # Strategy: Generate variable PR pattern with dissociation
        atrial_rate = random.randint(60, 100)  # Normal atrial rate
        ventricular_rate = random.randint(30, 50)  # Slower escape rhythm
        
        # Simplified: Generate pattern showing dissociation
        for i in range(num_beats):
            # Sometimes P-P pattern (atrial firing without capture)
            if random.random() < 0.3:
                sequence.extend(["P", "P"])
            # Sometimes P-R with variable PR
            elif random.random() < 0.6:
                sequence.append("P")
                # Variable PR (not constant, not consistently increasing)
                if i % 2 == 0:
                    sequence.append("PRlong")
                else:
                    sequence.append("PRincrease")
                sequence.append("R")
            # Sometimes R-R (ventricular escape)
            else:
                sequence.extend(["R", "R"])
        
        metadata = {
            "type": AVBlockType.THIRD_DEGREE.value,
            "atrial_rate_bpm": atrial_rate,
            "ventricular_rate_bpm": ventricular_rate,
            "description": "Complete AV dissociation with independent rhythms"
        }
        
        return sequence, metadata
    
    def generate_normal(self, num_beats: int = 5) -> Tuple[List[str], dict]:
        """
        Generate normal sinus rhythm pattern.
        
        Clinical characteristics:
        - PR interval 120-200ms (normal)
        - 1:1 AV conduction
        - Regular rhythm
        
        Args:
            num_beats: Number of cardiac cycles
        
        Returns:
            Tuple of (symbolic sequence, metadata)
        """
        sequence = []
        pr_duration = random.randint(120, 200)
        
        for i in range(num_beats):
            sequence.extend(["P", "PRnormal", "R"])
        
        metadata = {
            "type": AVBlockType.NORMAL.value,
            "pr_duration_ms": pr_duration,
            "num_beats": num_beats,
            "description": "Normal sinus rhythm"
        }
        
        return sequence, metadata
    
    def generate_dataset(self, samples_per_class: int = 20) -> List[dict]:
        """
        Generate a balanced dataset with all AV block types.
        
        Args:
            samples_per_class: Number of samples to generate per class
        
        Returns:
            List of samples, each containing sequence and metadata
        """
        dataset = []
        
        # First-Degree samples
        for i in range(samples_per_class):
            seq, meta = self.generate_first_degree(num_beats=random.randint(4, 7))
            dataset.append({
                "id": f"first_{i+1}",
                "sequence": seq,
                "label": AVBlockType.FIRST_DEGREE.value,
                "metadata": meta
            })
        
        # Mobitz I samples
        for i in range(samples_per_class):
            seq, meta = self.generate_mobitz_i(cycles=random.randint(1, 3))
            dataset.append({
                "id": f"mobitz_i_{i+1}",
                "sequence": seq,
                "label": AVBlockType.MOBITZ_I.value,
                "metadata": meta
            })
        
        # Mobitz II samples
        for i in range(samples_per_class):
            seq, meta = self.generate_mobitz_ii(
                num_conducted=random.randint(2, 4),
                num_blocks=random.randint(1, 3)
            )
            dataset.append({
                "id": f"mobitz_ii_{i+1}",
                "sequence": seq,
                "label": AVBlockType.MOBITZ_II.value,
                "metadata": meta
            })
        
        # Third-Degree samples
        for i in range(samples_per_class):
            seq, meta = self.generate_third_degree(num_beats=random.randint(5, 8))
            dataset.append({
                "id": f"third_degree_{i+1}",
                "sequence": seq,
                "label": AVBlockType.THIRD_DEGREE.value,
                "metadata": meta
            })
        
        # Normal samples
        for i in range(samples_per_class):
            seq, meta = self.generate_normal(num_beats=random.randint(4, 7))
            dataset.append({
                "id": f"normal_{i+1}",
                "sequence": seq,
                "label": AVBlockType.NORMAL.value,
                "metadata": meta
            })
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(self, dataset: List[dict], filename: str):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")
        print(f"Total samples: {len(dataset)}")
        
        # Print class distribution
        class_counts = {}
        for sample in dataset:
            label = sample['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("\nClass distribution:")
        for label, count in sorted(class_counts.items()):
            print(f"  {label}: {count} samples")


if __name__ == "__main__":
    # Generate dataset
    generator = SyntheticECGGenerator(seed=42)
    
    # Create dataset with 20 samples per class
    dataset = generator.generate_dataset(samples_per_class=20)
    
    # Save to file
    generator.save_dataset(dataset, "/home/claude/av_block_pda_project/data/synthetic_ecg_dataset.json")
    
    # Display example from each class
    print("\n" + "="*60)
    print("Example sequences:")
    print("="*60)
    
    seen_types = set()
    for sample in dataset:
        label = sample['label']
        if label not in seen_types:
            print(f"\n{label.upper()}:")
            print(f"  Sequence: {' '.join(sample['sequence'][:15])}...")
            print(f"  Description: {sample['metadata']['description']}")
            seen_types.add(label)
