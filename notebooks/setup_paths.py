import sys
from pathlib import Path

# Root projektu (o úroveň výš než notebooks/)
PROJECT_ROOT = Path(__file__).parent.parent

# Přidej src/ do Python path
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Definuj cesty k adresářům
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
SRC_DIR = PROJECT_ROOT / 'src'

# Pomocné cesty pro konkrétní soubory
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FIGURES_DIR = RESULTS_DIR / 'figures'
