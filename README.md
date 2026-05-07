# numberAI

A project to recreate machine learning from scratch without relying on pretrained models or pre-defined architectures.

## Overview

The core idea of **numberAI** is to categorize number images using simple matrix multiplication. Instead of starting with an established neural network architecture, the project utilizes a "trial and error" approach to discover the most suitable structure and matrix size for the task. This foundational approach aims to build a deep understanding of the mechanics underlying image classification.

## Project Phases

### Phase 1: Foundational Discovery
* **Goal**: Categorize number images using pure matrix multiplication.
* **Methodology**: Avoid pre-defined structures and pretrained weights. Experiment iteratively (trial and error) to determine optimal matrix dimensions and network topologies that yield accurate classification.

### Phase 2: Embedded Systems Deployment
* **Goal**: Run the classification model on resource-constrained embedded systems, such as an Arduino.
* **Methodology**: Apply model compression techniques, specifically **quantization** and **distillation**, to reduce the model's memory footprint and computational requirements while maintaining acceptable accuracy.