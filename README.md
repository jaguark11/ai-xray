# AI-XRay — Clasificación de radiografías de tórax

Kristian David Castrillón Paz  
Ciencia de Datos e IA Aplicada — DevSeniorCode

Clasificador de radiografías pediátricas en NORMAL y PNEUMONIA. Usa ResNet50, guarda los experimentos en MLflow y recibe imágenes mediante FastAPI. Es un ejercicio académico; no sirve para dar un diagnóstico médico.

## Modelos y datos

Este repositorio contiene el código, los notebooks ejecutados y los resultados. Las imágenes, los pesos y la base de MLflow se distribuyen en el paquete completo `AI-XRay_Kristian_Castrillon.zip`, como adjunto de una Release.

Antes de ejecutar, descargar ese adjunto desde **Releases** y extraerlo. Se puede trabajar directamente dentro de su carpeta `AI-XRay`, que contiene todo. Para usar una copia clonada de este repositorio, copiar desde el paquete completo `data/images/`, `models/aixray_model.keras`, `mlartifacts/` y `mlflow.db` a las mismas ubicaciones del repositorio.

Sin esos archivos, la API no puede cargar el modelo y las pruebas de inferencia no pueden ejecutarse. El ZIP automático **Source code** de GitHub solo incluye el código del repositorio.

## Cómo probarlo

Abrir una terminal en la carpeta del proyecto, una vez preparados los modelos y datos indicados arriba. Se necesita internet para instalar las dependencias.

Windows, con Python 3.12 de 64 bits:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_delivery.py
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/verify_delivery.py
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Abrir http://127.0.0.1:8000/docs. En `/predict`, pulsar **Try it out**, elegir un PNG de `data/images` y pulsar **Execute**. `/health` indica si el modelo está cargado. Las etiquetas originales están en `data/manifest_split.csv`.

Se probó en Linux con Python 3.12 y CPU. Windows tiene instrucciones, pero no se probó. El entorno exacto está en `requirements-lock-linux-py312.txt`.

## Datos y comparación

Se revisaron 5.856 imágenes y se preparó una muestra de 2.000: 1.389 para entrenar, 309 para validar y 302 para test. Antes de dividirlas se agruparon los pacientes identificables y los duplicados para evitar cruces entre conjuntos. La fuente y licencia están en `data/README.md`.

| Variante | Tasa de aprendizaje | Dropout | F1 validación | Recall | AUC |
|---|---:|---:|---:|---:|---:|
| A: base congelada | 0.0007 | 0.25 | 0.9527 | 0.9321 | 0.9913 |
| B: más dropout | 0.0003 | 0.5 | 0.9497 | 0.9321 | 0.9862 |
| C: ajuste final | 1e-05 | 0.25 | 0.9560 | 0.9383 | 0.9916 |

A y B entrenan el clasificador durante 8 épocas. B reduce la tasa y aumenta dropout para probar mayor regularización. Como cambian dos valores, no se puede atribuir la diferencia a uno solo. Ambas reutilizan características de la imagen original y una vista aumentada fija por imagen de entrenamiento.

C parte del mejor modelo congelado y ajusta `conv5_block3` durante 2 épocas, con nuevos aumentos por época y BatchNormalization fija. Se eligió por F1 de validación; los desempates usan recall y AUC. Test se reservó para evaluar al ganador, con umbral fijo de 0,5.

Resultados en test: **accuracy 96.36%, precision 97.35%, recall 95.45%, F1 96.39% y AUC 99.07%**. Hubo 4 falsas alarmas y 7 neumonías clasificadas como normales. Las predicciones y gráficas están en `results/test/`.

## Experimentos guardados

En los siguientes comandos, usar el `python` del entorno creado arriba.

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir http://127.0.0.1:5000 y buscar **AI-XRay-Kristian**. Hay tres ejecuciones con parámetros, métricas por época, gráficas, modelos y firmas de entrada y salida. El ganador es `finetune_conv5_block3_lr1e-5`, versión 1, en **Staging**, con alias `champion`.

Conservar juntos `mlflow.db` y `mlartifacts/`. Si se vuelve a entrenar, detener MLflow antes de ejecutar `python scripts/make_portable.py` y mover la carpeta.

## Archivos y ejecución completa

Los cuatro notebooks están en `notebooks/`, en orden: exploración, preparación, entrenamiento y evaluación. El código está en `src/` y `api/`; las pruebas y sus resultados en `tests/` y `results/`. El informe de tres páginas es `Executive_Summary_AI-XRay.docx`. En `docs/` están la consigna, la rúbrica y las opciones de uso.

Para repetir el entrenamiento y actualizar los documentos:

```bash
python run_pipeline.py
python scripts/verify_delivery.py
python scripts/execute_notebooks.py
python scripts/build_summary.py
python scripts/build_readme.py
python scripts/make_portable.py
```

La primera ejecución puede descargar pesos de ImageNet. Los notebooks revisan el entrenamiento guardado; poner `REENTRENAR=True` en el 03 para entrenar desde allí. PySpark necesita Java 17 y `pip install -r requirements-spark.txt`. Sin Spark, el notebook 01 lo avisa y muestra el resultado guardado.

## Alcance

La API acepta JPEG/PNG de hasta 10 MiB. `probability` es el score de PNEUMONIA, no una probabilidad clínica calibrada. Los errores y la configuración se explican en `docs/USO.md`.

No se observaron cruces por los identificadores y hashes disponibles, pero los nombres de archivo no permiten asegurar la identidad de todos los pacientes. No se probaron otros hospitales, edades o equipos. Los falsos negativos impiden usar el modelo para descartar neumonía.

La fase local incluye el paso de PySpark solicitado. En Databricks, las imágenes pasarían a un volumen, el manifiesto a Delta y los experimentos a MLflow gestionado; el entrenamiento podría usar GPU. Esa migración está planteada, no implementada, como permite la consigna.
