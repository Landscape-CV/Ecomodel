"""Island projects: many georeferenced tiles viewed and labeled as one cloud."""
from .store import Project, ProjectError, is_project_dir

__all__ = ["Project", "ProjectError", "is_project_dir"]
