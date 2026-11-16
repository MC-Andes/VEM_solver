# Contribuir

Seguir las [normas de MC-Andes](https://github.com/MC-Andes/standards). Trabajar
en ramas y proponer cambios mediante Pull Requests. Los cambios de formulación
deben explicar las hipótesis, unidades, tolerancias y evidencia numérica.

Instala las herramientas de desarrollo en tu entorno de Python:

```bash
python -m pip install -e '.[dev,docs]'
ruff check .
ruff format --check .
pytest
mkdocs build --strict
```

Adjunta el entorno y los resultados de las pruebas al Pull Request. Las
validaciones documentadas en esta migración se hicieron en Hypatia; la
[guía del clúster](docs/hypatia.md) explica ese procedimiento para quienes
tengan acceso. El desarrollo y el uso del paquete no dependen de ese servidor.

No versionar mallas grandes, temporales, credenciales ni resultados de producción.
Los datos pequeños de validación deben indicar su origen o cómo regenerarlos.
