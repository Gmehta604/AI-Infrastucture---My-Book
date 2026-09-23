# Gradient to Gigawatt

A field book on AI, from the first artificial neuron to the gigawatt data centers of 2026, written as local HTML pages. Deep dives into individual chapters are added over time, each with runnable code.

## Open the book

Double-click `index.html`. It opens in your browser; no server or install needed.

- Diagrams (Mermaid) and code colouring (highlight.js) load from jsDelivr the first time you open a page, so you need an internet connection for those. Text, tables and code are readable offline.
- "Mark chapter as read" and the theme choice are saved in your browser for this folder.

## Folder layout

```
gradient-to-gigawatt/
├── index.html            Home page and table of contents
├── README.md             This file
├── requirements.txt      Python packages used by the code/ folder
├── assets/
│   ├── book.css          Shared styles (light and dark theme)
│   └── book.js           Shared behaviour; also holds the chapter and deep-dive lists
├── chapters/
│   ├── _template.html    Template showing every page component
│   └── 01-story.html … 30-glossary.html
├── deep-dives/           Longer companion chapters (added one at a time)
└── code/
    ├── README.md         How to run the code
    └── NN…/              Scripts for each deep dive
```

## Adding a deep dive

1. Save the deep-dive page into `deep-dives/`.
2. Replace `assets/book.js` with the updated copy that comes with it (it adds one line to the `DEEP_DIVES` list).
3. Save its scripts into `code/`.
4. Refresh. The deep dive appears under its parent chapter on the home page, in the sidebar, and as a "Go deeper" link at the top of the chapter.

## Contents

| Part | Chapters |
|---|---|
| I · Foundations | 1 The Story of AI · 2 The Math You Actually Need · 3 Classical Machine Learning |
| II · Deep Learning | 4 Neural Networks and Backpropagation · 5 Training Deep Networks Well · 6 CNNs and Computer Vision · 7 Sequence Models |
| III · Transformers and LLMs | 8 Tokenization and Embeddings · 9 Attention and the Transformer · 10 Pretraining and Scaling Laws · 11 Mixture of Experts · 12 Post-Training · 13 Reasoning Models · 14 Multimodal and Generative Media |
| IV · AI Infrastructure | 15 GPUs and Accelerators · 16 CUDA, Triton and Kernels · 17 Distributed Training · 18 Fine-Tuning · 19 Inference and Serving · 20 LLM Gateways · 21 RAG · 22 Agents and MCP · 23 Evals and Observability · 24 The AI Platform · 25 Data Centers and Economics |
| V · Industry and Frontier | 26 The AI Industry Map · 27 Safety, Security and Policy · 28 The Frontier: September 2026 · 29 Your Path |
| Appendix | Glossary |
# AI-Infrastucture---My-Book
