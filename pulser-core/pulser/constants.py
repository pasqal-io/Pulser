from os import getenv

TRAP_WAVELENGTH = float(getenv("TRAP_WAVELENGTH", 0.85))  # µm
MASS = float(getenv("MASS", 1.45e-25))  # kg
KB = float(getenv("KB", 1.38e-23))  # J/K
KEFF = float(getenv("KEFF", 8.7))  # µm^-1
