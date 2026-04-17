# Evolución de los Modelos de Lenguaje

Una guía conceptual y práctica sobre cómo hemos pasado de los n-gramas a los Large Language Models (LLMs), y por qué el **scaling** cambió la manera de hacer NLP.

> Lectura complementaria al notebook `02_modelos_lenguaje.ipynb`.

---

## Tabla de contenidos

1. [¿Qué es un modelo de lenguaje?](#1-qué-es-un-modelo-de-lenguaje)
2. [SLM — Statistical Language Models](#2-slm--statistical-language-models)
3. [NLM — Neural Language Models](#3-nlm--neural-language-models)
4. [PLM — Pre-trained Language Models](#4-plm--pre-trained-language-models)
5. [LLM — Large Language Models](#5-llm--large-language-models)
6. [Leyes de escalamiento (Scaling Laws)](#6-leyes-de-escalamiento-scaling-laws)
7. [Recursos recomendados](#7-recursos-recomendados)

---

## 1. ¿Qué es un modelo de lenguaje?

Un **modelo de lenguaje (LM)** asigna una probabilidad a una secuencia de palabras (o tokens):

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1})
$$

Dicho de otra forma: dado lo que vimos antes, ¿qué tan probable es el siguiente token? Esta idea aparentemente simple es la base desde Shannon (1948) hasta GPT‑4.

Todas las generaciones de modelos que veremos (SLM → NLM → PLM → LLM) atacan el mismo problema con herramientas cada vez más potentes.

---

## 2. SLM — Statistical Language Models

Los **Statistical Language Models** son la primera generación: usan frecuencias observadas en un corpus para estimar probabilidades.

### 2.1 N-gramas

Por la **regla de la cadena** la probabilidad de una secuencia depende de todo el pasado. Esto es intratable, así que aplicamos la **suposición de Markov**: solo importan los últimos `n−1` tokens.

$$
P(w_t \mid w_1, \dots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \dots, w_{t-1})
$$

- **Unigrama** (n=1): cada palabra es independiente.
- **Bigrama** (n=2): cada palabra depende de la anterior.
- **Trigrama** (n=3): depende de las dos anteriores.

La estimación por **máxima verosimilitud** es simple conteo:

$$
P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}
$$

### 2.2 Problema: data sparsity

Si un bigrama no aparece en el corpus, su probabilidad es **cero** — y por el producto, toda la frase se vuelve imposible. Dado que el lenguaje es combinatoriamente enorme, esto pasa todo el tiempo.

### 2.3 Suavizado (smoothing)

Técnicas para asignar masa de probabilidad a n‑gramas no vistos:

- **Laplace / add‑k**: suma una constante pequeña a todos los conteos.
- **Good‑Turing**: reasigna probabilidad de eventos vistos una vez a eventos no vistos.
- **Kneser‑Ney**: considera **versatilidad** contextual de una palabra (estado del arte clásico).
- **Back‑off** e **interpolación**: si el n‑grama es raro, consulta modelos de menor orden.

### 2.4 Limitaciones de los SLM

- **Ventana fija**: solo capturan el contexto local (2–5 tokens).
- **Representación discreta**: "perro" y "can" son tokens distintos sin relación.
- **Escalabilidad**: el número de n‑gramas posibles crece exponencialmente con `n`.
- **Sin generalización semántica**: no entienden sinónimos ni contexto largo.

### Lecturas

- Jurafsky & Martin — *Speech and Language Processing*, Cap. 3 ["N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- Chen & Goodman (1999) — ["An Empirical Study of Smoothing Techniques for Language Modeling"](https://aclanthology.org/J99-3005/)

---

## 3. NLM — Neural Language Models

La segunda generación introduce **redes neuronales** y, sobre todo, **representaciones densas** (embeddings) que capturan semántica.

### 3.1 Bengio et al. (2003) — el primer NLM

El modelo "A Neural Probabilistic Language Model" de Bengio aprende simultáneamente:

1. Un **embedding** por palabra (vector denso de baja dimensión).
2. Un modelo de lenguaje que predice la siguiente palabra a partir de esos embeddings.

Esto resuelve el problema de la dispersión: palabras semánticamente parecidas terminan con vectores parecidos, y el modelo **generaliza** a combinaciones nunca vistas.

### 3.2 Word2Vec (Mikolov et al., 2013)

Mikolov mostró que se puede aprender embeddings de calidad con arquitecturas **muy eficientes**:

- **CBOW** (Continuous Bag‑of‑Words): predice la palabra central dado su contexto.
- **Skip‑gram**: dada una palabra central, predice las palabras del contexto.

Los embeddings resultantes mostraron propiedades **composicionales** famosas:

$$
\text{vec}(\text{rey}) - \text{vec}(\text{hombre}) + \text{vec}(\text{mujer}) \approx \text{vec}(\text{reina})
$$

### 3.3 GloVe y FastText

- **GloVe** (Pennington et al., 2014): combina estadísticas globales (co‑ocurrencias) con lo local.
- **FastText** (Bojanowski et al., 2016): usa **subpalabras** (n‑gramas de caracteres), lo que permite manejar palabras fuera del vocabulario y morfología rica (ideal para el español).

### 3.4 Modelos neuronales secuenciales

Con embeddings listos, aparecen los LMs neuronales de **contexto largo**:

- **RNN** (Elman, 1990): mantienen un estado oculto que recorre la secuencia.
- **LSTM** (Hochreiter & Schmidhuber, 1997): añaden **compuertas** que controlan qué recordar y qué olvidar, resolviendo el problema del gradiente que desaparece.
- **GRU** (Cho et al., 2014): versión simplificada de LSTM.

Estos modelos fueron estado del arte en NLP entre ~2014 y 2018.

### 3.5 Limitaciones de los NLM

- **Secuenciales**: difíciles de paralelizar en GPU.
- **Contexto efectivo limitado** (aunque mayor que n‑gramas).
- **Embeddings estáticos** (en word2vec/GloVe): "banco" tiene el mismo vector tanto para la institución financiera como para el mueble.

### Lecturas

- Bengio et al. (2003) — ["A Neural Probabilistic Language Model"](https://www.jmlr.org/papers/v3/bengio03a.html)
- Mikolov et al. (2013) — ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)
- Mikolov et al. (2013) — ["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546)
- Pennington et al. (2014) — ["GloVe"](https://aclanthology.org/D14-1162/)
- Hochreiter & Schmidhuber (1997) — ["Long Short‑Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf)

---

## 4. PLM — Pre-trained Language Models

La tercera generación introduce el paradigma **pre‑training + fine‑tuning**: entrenamos un modelo enorme con tarea auto‑supervisada sobre mucho texto, y luego lo adaptamos a tareas concretas con poco data.

Es el cambio más importante del NLP moderno: pasamos de **feature engineering por tarea** a **representaciones universales**.

### 4.1 ELMo (Peters et al., 2018)

- Embeddings **contextuales**: el vector de "banco" cambia según la oración.
- Obtenidos de una **biLSTM** profunda entrenada como modelo de lenguaje bidireccional.
- Se usa como feature input a modelos específicos de tarea.

### 4.2 Transformer (Vaswani et al., 2017)

El artículo *"Attention Is All You Need"* propone eliminar la recurrencia y apoyarse en **self‑attention**:

- Cada token atiende a todos los demás en paralelo.
- Permite contextos largos y entrenamiento masivamente paralelo en GPU/TPU.

Es la base arquitectónica de todo lo que viene después.

### 4.3 BERT (Devlin et al., 2018) — encoder bidireccional

- Arquitectura: **encoder** del Transformer.
- Pre‑training con dos tareas:
  - **Masked Language Modeling (MLM)**: predecir palabras enmascaradas usando contexto a ambos lados.
  - **Next Sentence Prediction (NSP)**: ¿estas dos oraciones van seguidas?
- Fine‑tuning para tareas supervisadas: clasificación, NER, QA, etc.
- Variantes: **RoBERTa**, **ALBERT**, **DistilBERT**, **BETO** (español), **DeBERTa**.

### 4.4 GPT (Radford et al., 2018–2019) — decoder autoregresivo

- Arquitectura: **decoder** del Transformer (causal, solo mira hacia la izquierda).
- Pre‑training: predecir el siguiente token.
- Fine‑tuning inicial para tareas, pero ya se observa **zero‑shot** emergiendo en GPT‑2.
- Genera texto de manera natural porque su objetivo es literalmente "continuar la secuencia".

### 4.5 T5, BART, XLNet, ELECTRA

La era PLM produjo una explosión de variantes. Algunos hitos:

- **T5** (Raffel et al., 2020): todo es texto‑a‑texto.
- **BART**: seq2seq con denoising.
- **ELECTRA**: más eficiente reemplazando MLM por detección.

### 4.6 Por qué importa el paradigma PLM

- **Transfer learning**: pre‑entrenas una vez sobre miles de millones de palabras y luego fine‑tuneas con miles de ejemplos.
- **Universalidad**: el mismo modelo resuelve clasificación, NER, QA, similitud, resumen.
- **Contextualización**: los embeddings ya no son estáticos.

### Lecturas

- Vaswani et al. (2017) — ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Peters et al. (2018) — ["Deep Contextualized Word Representations (ELMo)"](https://arxiv.org/abs/1802.05365)
- Devlin et al. (2018) — ["BERT"](https://arxiv.org/abs/1810.04805)
- Radford et al. (2019) — ["Language Models are Unsupervised Multitask Learners (GPT‑2)"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- Jay Alammar — ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
- Jay Alammar — ["The Illustrated BERT, ELMo & co."](https://jalammar.github.io/illustrated-bert/)

---

## 5. LLM — Large Language Models

La cuarta generación es, en esencia, PLM **pero a una escala que revela fenómenos nuevos**.

### 5.1 Qué define a un LLM

No hay una definición formal, pero se suele incluir:

- **Parámetros**: >10B (usualmente cientos de miles de millones).
- **Datos de entrenamiento**: cientos de B → T de tokens.
- **Cómputo**: típicamente >10²³ FLOPs.
- **Capacidades**: generalización zero/few‑shot sin fine‑tuning.

### 5.2 Emergent abilities

A partir de cierto tamaño aparecen capacidades que **no existen** en modelos más pequeños ni se predicen suavemente: aritmética multi‑paso, razonamiento encadenado (chain‑of‑thought), seguimiento de instrucciones, traducción zero‑shot, etc.

> Wei et al. (2022) documentan decenas de estas transiciones abruptas. El debate sigue: algunos trabajos posteriores (Schaeffer et al., 2023) argumentan que son artefacto de métricas discretas — pero el fenómeno práctico persiste.

### 5.3 In-context learning (ICL)

Es quizá la propiedad más disruptiva: el modelo **aprende de ejemplos en el prompt** sin actualizar sus pesos.

- **Zero‑shot**: le pides la tarea directamente.
- **Few‑shot**: le das 1–n ejemplos antes de la consulta real.
- **Chain‑of‑thought**: le pides que razone paso a paso.

Esto cambió el paradigma: en vez de `fine‑tune a new model per task`, ahora es `prompt the same model`.

### 5.4 Modelos representativos

| Modelo | Año | Parámetros | Notas |
|--------|-----|------------|-------|
| GPT‑3 | 2020 | 175B | El artículo que popularizó few‑shot learning. |
| Chinchilla | 2022 | 70B | Menos params, más tokens — cambió el "recipe". |
| PaLM | 2022 | 540B | Escala masiva, razonamiento. |
| LLaMA / LLaMA 2 / 3 | 2023–24 | 7B–70B+ | Open weights, llevaron LLMs al público. |
| GPT‑4 | 2023 | ? (no público) | Multimodal, fuerte razonamiento. |
| Mistral, Mixtral | 2023–24 | 7B–8×22B | MoE competitivo en abierto. |
| Claude 3/4 | 2024–25 | ? | Anthropic, foco en seguridad y razonamiento. |

### 5.5 Post‑entrenamiento

Un LLM moderno no es solo el pre‑training:

- **SFT** (Supervised Fine‑Tuning): se le enseña a seguir instrucciones.
- **RLHF / DPO**: alineación con preferencias humanas.
- **Tool use / Agents**: integración con herramientas externas.

### Lecturas

- Brown et al. (2020) — ["Language Models are Few‑Shot Learners (GPT‑3)"](https://arxiv.org/abs/2005.14165)
- Wei et al. (2022) — ["Emergent Abilities of Large Language Models"](https://arxiv.org/abs/2206.07682)
- Wei et al. (2022) — ["Chain‑of‑Thought Prompting"](https://arxiv.org/abs/2201.11903)
- Schaeffer et al. (2023) — ["Are Emergent Abilities a Mirage?"](https://arxiv.org/abs/2304.15004)
- Zhao et al. (2023) — ["A Survey of Large Language Models"](https://arxiv.org/abs/2303.18223) (referencia completísima)
- Ouyang et al. (2022) — ["InstructGPT / RLHF"](https://arxiv.org/abs/2203.02155)

---

## 6. Leyes de escalamiento (Scaling Laws)

Una pregunta práctica crítica: si tengo un presupuesto de cómputo `C`, ¿cuántos parámetros `N` y cuántos tokens de entrenamiento `D` debo usar?

### 6.1 Ley de Kaplan‑McCandlish (KM, 2020)

OpenAI publicó *"Scaling Laws for Neural Language Models"* mostrando que la **cross‑entropy** de un LM sigue **leyes de potencia** frente a `N`, `D` y `C`:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

Conclusión principal de Kaplan et al.: dado un presupuesto fijo, es **más eficiente hacer el modelo más grande** que entrenarlo más tiempo. Esta receta guió el diseño de GPT‑3 (175B con relativamente pocos tokens).

### 6.2 Ley de Chinchilla (Hoffmann et al., 2022)

DeepMind re‑examinó la cuestión en *"Training Compute‑Optimal Large Language Models"* y encontró un resultado muy distinto:

> Para ser **compute‑optimal**, `N` y `D` deben escalar **proporcionalmente**.

La regla práctica famosa: **~20 tokens por cada parámetro**.

Lo demostraron entrenando **Chinchilla (70B)** con ~1.4T tokens y superando a **Gopher (280B)** entrenado con solo ~300B tokens — usando **el mismo cómputo**.

### 6.3 ¿KM vs Chinchilla?

No es que Kaplan estuviera "mal": usó un grid de hiperparámetros (en particular el scheduler de learning rate) que sub‑optimizaba los modelos entrenados con muchos tokens, lo que sesgó la conclusión. Chinchilla corrigió esto y la nueva receta fue adoptada por todos: LLaMA, Mistral, etc., se entrenan con miles de tokens por parámetro.

### 6.4 Después de Chinchilla

- **Over‑training**: LLaMA 3 se entrena con ~15T tokens en 8B parámetros (>>20 tokens/parámetro). Razón: el cómputo de inferencia es mayor que el de entrenamiento cuando el modelo se desplegará masivamente, así que sobre‑entrenar modelos pequeños es económicamente óptimo.
- **Scaling laws para RLHF, datos sintéticos, mezcla de expertos (MoE)** y más son áreas activas de investigación.

### 6.5 Qué te llevas

1. Cross‑entropy escala predeciblemente con `N`, `D` y `C`.
2. Para entrenamiento compute‑optimal: ~20 tokens por parámetro (Chinchilla).
3. Para **inferencia‑optimal**: entrena modelos más pequeños con muchísimos más tokens.
4. Los scaling laws son el motor detrás de *"si duplico el cómputo, sé cuánto mejor será mi modelo"* — clave para la inversión y el diseño.

### Lecturas

- Kaplan et al. (2020) — ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361)
- Hoffmann et al. (2022) — ["Training Compute‑Optimal Large Language Models (Chinchilla)"](https://arxiv.org/abs/2203.15556)
- Henighan et al. (2020) — ["Scaling Laws for Autoregressive Generative Modeling"](https://arxiv.org/abs/2010.14701)
- Hoffmann et al. (2022) — DeepMind blog ["An empirical analysis of compute‑optimal LLM training"](https://deepmind.google/discover/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)

---

## 7. Recursos recomendados

### Libros y cursos

- **Jurafsky & Martin** — *Speech and Language Processing* (3ª ed.), gratis en [stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)
- **Stanford CS224N** — NLP with Deep Learning ([curso](http://web.stanford.edu/class/cs224n/) · [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ))
- **Hugging Face NLP Course** — [huggingface.co/learn/nlp-course](https://huggingface.co/learn/nlp-course)

### Surveys y timelines

- Zhao et al. (2023) — ["A Survey of Large Language Models"](https://arxiv.org/abs/2303.18223)
- Minaee et al. (2024) — ["Large Language Models: A Survey"](https://arxiv.org/abs/2402.06196)
- Hugging Face — ["The Evolution of Language Models"](https://huggingface.co/blog/evolution-of-ml-language-models)

### Visualizaciones

- Jay Alammar — [The Illustrated Transformer / BERT / GPT2](https://jalammar.github.io/)
- Andrej Karpathy — ["Let's build GPT: from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video)
- Andrej Karpathy — ["State of GPT"](https://www.youtube.com/watch?v=bZQun8Y4L2A) (charla)

### Papers curados

- Sebastian Ruder — ["NLP progress"](http://nlpprogress.com/)
- "Awesome LLM" GitHub — [github.com/Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)

---

## De este repositorio

- Notebook ejecutable con los conceptos de este documento: `02_modelos_lenguaje.ipynb`
- Notebook con deep learning aplicado a NLP (RNN, LSTM, Autoencoder) usando *El Quijote*: `../03_deep_learning_nlp/03_deep_learning_nlp.ipynb`
- Notebook introductorio (clásico): `../01_intro_nlp/01_introduccion_nlp.ipynb`
