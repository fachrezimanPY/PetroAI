APP_NAME = "PetroAI"
APP_VERSION = "1.0.0"

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"  # ganti sesuai model yang terinstall

# UI settings
THEME = "dark"
LOG_TRACK_WIDTH = 120

# Petrophysics defaults
GR_SAND = 15.0    # GR clean sand (API)
GR_SHALE = 120.0  # GR shale (API)
RW = 0.05         # Formation water resistivity (ohm.m)
A = 1.0           # Archie tortuosity factor
M = 2.0           # Archie cementation exponent
N = 2.0           # Archie saturation exponent
