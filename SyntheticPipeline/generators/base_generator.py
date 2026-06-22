from abc import ABC, abstractmethod

class BaseTreeGenerator(ABC):
    """
    Abstract base class for tree generators in the Synthetic Pipeline.
    Any new tree generation backend (e.g., L-Systems, AI models, custom procedural tools)
    must implement this interface to be compatible with the pipeline.
    """

    @abstractmethod
    def generate_tree(self, seed: int, height: float, out_obj: str, out_json: str, **kwargs):
        """
        Generates a tree asset and its corresponding ground truth skeleton.

        Args:
            seed (int): Random seed for reproducible generation.
            height (float): Target height scale for the tree.
            out_obj (str): Filepath to save the exported .obj model (leaf-on).
                           The generator should also ideally produce a `{out_obj}_noleaf.obj` file.
            out_json (str): Filepath to save the exported skeleton data as .json.
            **kwargs: Additional generator-specific arguments (e.g., config paths, executables).
        """
        pass
