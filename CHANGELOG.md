# Changelog

## 0.1.0 — preparación local, 2026-11-16

- Separación del núcleo monolítico en módulos por responsabilidad y CLI instalable.
- Desplazamientos CSR calculados en int64; índices de salida de 64 bits cuando
  el número de tripletas lo requiere.
- Matrices vacías sin memmap de tamaño cero, cierre de archivos ante fallos.
- Verificación explícita de convergencia AMG y limpieza de sistemas fallidos.
- Orientación de caras mediante IDs de nodo, independiente de la numeración GDL.
- Pruebas numéricas, compatibilidad histórica y trabajos Slurm para Hypatia.
- Estabilización con productos de rango 12, proyección compacta y reducción de
  campos cuando sólo se necesitan propiedades efectivas.
- PyAMG mínimo 5.4.1 para kernels int64; diagnóstico explícito en versiones antiguas.
- Recomendación automática de procesos limitada por la asignación Slurm.
- Inicio afín de AMG, con verificación del mismo residuo y opción `--inicio-cero`.
  La equivalencia se verifica en medios homogéneos y heterogéneos.
