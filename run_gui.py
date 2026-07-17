import sys
from pathlib import Path

# Add the project root to sys.path so `from ecomodel import Ecomodel` and
# `from gui.xxx import ...` both resolve correctly, regardless of the
# working directory from which this script is launched.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from gui.main_window import EcomodelMainWindow

# App icon shown in the macOS Dock / Windows-Linux taskbar at launch.
_ICON = _ROOT / "gui" / "assets" / "ecomodel_E_1024.png"


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Ecomodel")
    app.setOrganizationName("PyTLidar")
    if _ICON.exists():
        app.setWindowIcon(QIcon(str(_ICON)))

    window = EcomodelMainWindow()
    if _ICON.exists():
        window.setWindowIcon(QIcon(str(_ICON)))
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
