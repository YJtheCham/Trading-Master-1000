import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from streamlit.web import cli as stcli
import sys

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", str(Path(__file__).parent / "app.py")]
    sys.exit(stcli.main())
