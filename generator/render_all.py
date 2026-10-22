import sys
from engine import render
from scenes import SCENES
names = {"retrowave": "Неоновый закат", "gradients": "Градиенты", "nebula": "Туманность",
         "steppe": "Звёзды над степью", "aurora": "Северное сияние", "canyon": "Ущелье"}
for key in sys.argv[1:]:
    cls, sec = SCENES[key]
    render(cls(), f"out/{names[key]}.mov", sec, fps=60 if key == "canyon" else 30)
