import warnings

def run_adqsm_on_segments(*args, **kwargs):
    """
    Placeholder wrapper for AdQSM.
    AdQSM does not support a native Command-Line Interface (CLI) or batch processing.
    It is a standalone GUI tool where point clouds must be manually loaded and processed.
    Therefore, it cannot be run automatically within the benchmark pipeline.
    """
    warnings.warn("AdQSM cannot be run automatically because it lacks CLI support. Please run AdQSM manually using the GUI executable in thirdparty/AdQSM.")
    return None
