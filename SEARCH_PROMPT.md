# Prompt para Buscar Papers Similares a BILP

## Prompt para GPT/Claude/Perplexity:

```
Busca papers académicos y artículos científicos que describan métodos similares al siguiente enfoque para Person Re-Identification:

**Método BILP (Brightness-Invariant Local Patterns):**

1. **Características Handcrafted (no deep learning):**
   - Features de color invariantes a brillo usando log-chromaticity (u = log(R/G), v = log(B/G))
   - Features de textura usando filtros Gabor multi-escala (5 escalas × 8 orientaciones) y análisis FFT
   - División de imágenes en stripes horizontales (6 stripes) para capturar información espacial
   - Histogramas de color (16×16 bins para u,v + 16 bins para luminancia) por stripe
   - Energía de respuesta Gabor + características espectrales (frecuencia pico, entropía) por stripe

2. **Fusión Adaptativa:**
   - Gating adaptativo para combinar features de color y textura: d = α·d_texture + (1-α)·d_color
   - El peso α se calcula dinámicamente basado en contenido de la imagen (entropía cromática y energía de alta frecuencia)
   - Normalización L1 por stripe antes de la fusión

3. **Evaluación:**
   - Datasets: Market-1501 e iLIDS-VID
   - Métricas: mAP (mean Average Precision) y CMC (Cumulative Matching Characteristics)
   - Comparación con baseline de deep learning (ResNet50)

**Búsqueda específica:**
- Papers que combinen color invariante a iluminación (log-chromaticity, opponent color space) con textura (Gabor, LBP, o similar) para Person Re-ID
- Métodos handcrafted que usen gating adaptativo o fusión dinámica de múltiples tipos de features
- Trabajos que evalúen en Market-1501 o iLIDS-VID con métodos no-deep-learning
- Papers sobre "brightness-invariant" o "illumination-invariant" features para re-identification
- Métodos que dividan imágenes en stripes o regiones horizontales para extracción de features

**Excluir:**
- Métodos puramente basados en deep learning (CNN, ResNet, etc.) sin componentes handcrafted
- Trabajos que solo usen deep learning pre-entrenado sin features handcrafted

**Formato de respuesta:**
Para cada paper encontrado, proporciona:
1. Título completo
2. Autores y año
3. Conferencia/Journal
4. Resumen breve (2-3 líneas) explicando las similitudes con BILP
5. Link/DOI si está disponible

Busca en bases de datos como: arXiv, IEEE Xplore, ACM Digital Library, Google Scholar, CVPR, ICCV, ECCV, BMVC, WACV.
```

## Variante más específica (si la primera no da resultados):

```
Busca papers sobre métodos handcrafted para Person Re-Identification que:

1. Usen log-chromaticity o color invariante a iluminación (u, v channels)
2. Combinen color con textura usando filtros Gabor o análisis espectral
3. Implementen fusión adaptativa o gating para combinar múltiples tipos de features
4. Evalúen en Market-1501 o iLIDS-VID
5. Reporten resultados usando mAP y CMC metrics

Keywords específicos a buscar:
- "log-chromaticity" AND "person re-identification"
- "Gabor filters" AND "person re-id" AND "handcrafted"
- "adaptive fusion" AND "person re-identification" AND "color texture"
- "brightness invariant" AND "re-identification"
- "horizontal stripes" AND "person re-id" AND "features"

Excluir papers que sean:
- Puramente deep learning sin componentes handcrafted
- Solo usen deep learning pre-entrenado sin diseño de features manual
```

## Variante para buscar implementaciones similares:

```
Busca implementaciones de código abierto o repositorios GitHub que implementen métodos similares a:

- Person Re-Identification usando features handcrafted (no deep learning)
- Combinación de color invariante (log-chromaticity) y textura (Gabor filters)
- Fusión adaptativa de múltiples tipos de features
- Evaluación en Market-1501 o iLIDS-VID

Incluye:
- Repositorios GitHub con código Python
- Implementaciones que usen OpenCV, scikit-image, o librerías similares
- Proyectos que combinen múltiples tipos de features handcrafted
```

