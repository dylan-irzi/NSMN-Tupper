# NSMN-Tupper: Active Vision & Holographic Codebooks

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**NSMN-Tupper** (Neural Saccadic Memory Network) es una arquitectura de visión computacional ultra-eficiente que combina **mecanismos de sacadas biológicas** con la **Fórmula Autoreferencial de Tupper** para lograr un rendimiento competitivo con una fracción de los parámetros de las redes convencionales.

## 🚀 Resumen del Desempeño
| Modelo | Parámetros | Épocas | CIFAR-10 (Acc@1) |
| :--- | :--- | :--- | :--- |
| ResNet-18 | 11.2M | 200 | ~93.0% |
| **NSMN-Tupper** | **3.5M** | **120 (93% @ 65ep)** | **93.2%** |

## 🧠 ¿Qué hace único a este modelo?

A diferencia de las CNN convencionales que procesan la imagen de forma estática, NSMN-Tupper utiliza tres pilares innovadores:

1.  **Visión Activa (Saccadic Glimpses):** El modelo no "mira" toda la imagen a la vez. Utiliza una red de localización que selecciona secuencialmente las zonas más informativas (glimpses) a diferentes escalas, emulando el movimiento ocular humano.
2.  **Memoria Recurrente (GRU):** La información extraída de cada "sacada" se integra en una unidad de memoria recurrente (GRU), permitiendo que el modelo refine su predicción con cada vistazo.
3.  **Codebook Holográfico de Tupper:** En lugar de clasificar solo mediante un vector *one-hot*, el modelo debe "reconstruir" un patrón binario derivado de la **Fórmula Autoreferencial de Tupper**. Esto actúa como un regularizador de alta entropía que previene el sobreajuste y obliga a la red a aprender características estructurales profundas.

## 🏗️ Arquitectura

El flujo de datos se divide en:
- **ContextNet:** Una ResNet ligera que genera el mapa de características inicial.
- **LocationNetwork:** Predice las coordenadas $(x, y)$ del siguiente vistazo.
- **GlimpseNet:** Extrae parches multiescala mediante transformaciones afines.
- **Holographic Head:** Proyecta la memoria final hacia el espacio de bits de Tupper para la clasificación por similitud de coseno.

```python
# La magia de la eficiencia
loss = loss_cross_entropy + 2.0 * loss_tupper_reconstruction
```

## 🛠️ Instalación y Uso

### Requisitos
- Python 3.8+
- PyTorch (CUDA compatible recomendado)
- Torchvision
- Matplotlib

### Ejecución
Para entrenar el modelo desde cero:
```bash
python main.py
```

## 📊 Visualización de Resultados

El modelo no solo predice la clase, sino que "imagina" el código de Tupper asociado:

| Imagen Original | Tupper Generado (Predicción) | Objetivo (Codebook) |
| :---: | :---: | :---: |
| ![Car](https://via.placeholder.com/100?text=Car) | ![Gen](https://via.placeholder.com/100?text=Bits) | ![Target](https://via.placeholder.com/100?text=Tupper) |

*Nota: Durante el entrenamiento, el modelo aprende a mapear las características visuales hacia la constante de Tupper de forma determinista.*

## 📈 Curva de Aprendizaje
Gracias al optimizador `OneCycleLR` y al codebook fijo, el modelo presenta una convergencia extremadamente rápida, alcanzando el **90% de precisión en menos de 40 épocas**.

## 📄 Licencia
Este proyecto está bajo la Licencia MIT.

## 🤝 Créditos
Desarrollado como una exploración en arquitecturas de visión eficientes y regularización geométrica. Inspirado en el trabajo de Kaiming He (ResNet) y Jeff Tupper (Tupper's Self-Referential Formula).
