import os
import argparse
import multiprocessing
import random
from pathlib import Path
import sys

# Import generators
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generators.base_generator import BaseTreeGenerator

def run_generator_task(args):
    """
    Wrapper to run the generation task in a separate process.
    """
    generator, seed, height, out_obj, out_json = args
    generator.generate_tree(seed, height, out_obj, out_json)

class AssetManager:
    """
    Manages the generation of tree assets utilizing a provided Tree Generator.
    Supports multiprocessing for parallel generation.
    """
    def __init__(self, generator: BaseTreeGenerator, output_dir: str = "SyntheticPipeline/output/assets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generator = generator

    def generate_trees(
        self,
        num_trees: int = 10,
        height_range: tuple = (8.0, 15.0),
        num_workers: int = 4,
        name_prefix: str = "tree",
        start_index: int = 0,
    ):
        """
        Generates a specified number of trees in parallel.
        
        Args:
            num_trees (int): Number of trees to generate.
            height_range (tuple): Min and max height for trees.
            num_workers (int): Number of parallel processes to use.
            name_prefix (str): Filename prefix (species name for multi-species pools).
            start_index (int): Starting index for filenames.
        """
        tasks = []
        for i in range(start_index, start_index + num_trees):
            seed = random.randint(0, 999999)
            height = random.uniform(*height_range)
            out_obj = str(self.output_dir / f"{name_prefix}_{i:04d}.obj")
            out_json = str(self.output_dir / f"{name_prefix}_{i:04d}.json")
            
            # Skip if already generated
            if os.path.exists(out_obj) and os.path.exists(out_json):
                print(f"{name_prefix}_{i:04d} already exists, skipping...")
                continue
                
            tasks.append((self.generator, seed, height, out_obj, out_json))
            
        if not tasks:
            print("No new trees to generate.")
            return

        print(f"Starting generation of {len(tasks)} trees using {num_workers} workers...")
        with multiprocessing.Pool(num_workers) as pool:
            pool.map(run_generator_task, tasks)
        print("Asset generation complete.")
