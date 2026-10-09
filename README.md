# Simulador web de intercambiador de calor

Aplicación académica en Streamlit, basada en el modelo original de intercambiador de doble tubo en contracorriente.

## Publicar y compartir un enlace

1. Crea un repositorio en GitHub y sube `app.py`, `modelo.py` y `requirements.txt` a la raíz.
2. Entra a https://share.streamlit.io y accede con GitHub.
3. Pulsa **Create app** y elige el repositorio, la rama `main` y el archivo `app.py`.
4. Pulsa **Deploy** y espera a que el sitio esté publicado.
5. Comparte el enlace `https://...streamlit.app` que te muestre Streamlit.

## Ejecución local opcional

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Consideraciones

- La primera predicción de Random Forest puede tardar porque se generan 5000 muestras y se entrena un bosque de 500 árboles. Los resultados se guardan temporalmente en caché.
- La aplicación conserva las funciones y parámetros originales, incluido el criterio `or` de interrupción del método de bisección.
- Las predicciones de Random Forest solo se presentan dentro de los rangos de entrenamiento.
- La precisión se evalúa frente a escenarios sintéticos, no contra mediciones de una planta industrial.
