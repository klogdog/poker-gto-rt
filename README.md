# Poker GTO-RT - Real-Time Poker Analysis

**Real-time poker analysis system with computer vision and GTO solver. Optimized for < 400ms latency on Apple Silicon.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![CoreML](https://img.shields.io/badge/CoreML-Apple%20Silicon-purple.svg)](https://developer.apple.com/machine-learning/core-ml/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Overview

Poker GTO-RT is a real-time poker analysis system that combines computer vision, OCR, and Game Theory Optimal (GTO) solving. The system analyzes poker tables in real-time and provides optimal decision support.

**Key Highlights** :
- ✅ **Computer Vision** : OCR, YOLO, SAM2 for table/card detection
- ✅ **GTO Solver** : CFR++ and Monte-Carlo methods
- ✅ **Real-Time** : < 400ms latency on Apple Silicon
- ✅ **CoreML Optimized** : Native Apple Silicon performance

---

## 🛠️ Technologies

- **Python** 3.11+
- **CoreML** (Apple Silicon optimization)
- **YOLO** (Object detection)
- **SAM2** (Segment Anything Model)
- **OCR** (Text recognition, calibrated for poker tables)
- **CFR++** (Counterfactual Regret Minimization)
- **Monte-Carlo** (Simulation methods)

---

## 📊 Architecture

```
┌──────────────┐
│  Image Input │  (Poker table screenshot)
└──────┬───────┘
       │
┌──────▼───────┐
│  Vision      │  (YOLO: table/cards detection)
└──────┬───────┘
       │
┌──────▼───────┐
│  OCR         │  (Text recognition: pot, bets)
└──────┬───────┘
       │
┌──────▼───────┐
│  GTO Solver  │  (CFR++, Monte-Carlo)
└──────┬───────┘
       │
┌──────▼───────┐
│  Decision    │  (Optimal action recommendation)
└──────────────┘
```

---

## 🚀 Key Features

### **1. Computer Vision**
- **YOLO** : Detects poker table, cards, and player positions
- **SAM2** : Precise segmentation of table elements
- **OCR** : Calibrated text recognition for pot size, bets, stack sizes
- **Calibration** : System calibrated specifically for poker tables

### **2. GTO Solver**
- **CFR++** : Counterfactual Regret Minimization for optimal strategy
- **Monte-Carlo** : Simulation-based decision support
- **Real-Time** : Fast enough for live play analysis

### **3. Apple Silicon Optimization**
- **CoreML** : Native Apple Silicon performance
- **Latency** : < 400ms end-to-end processing
- **M4 Max** : Optimized for 36GB RAM

---

## 📈 Performance

- **Latency** : < 400ms end-to-end
- **Accuracy** : High accuracy OCR for poker tables
- **Real-Time** : Suitable for live analysis

---

## 📝 Status

**Phase** : Completed - Functional system

**Completed** :
- ✅ Complete vision pipeline (OCR, YOLO, SAM2)
- ✅ GTO solver (CFR++, Monte-Carlo)
- ✅ Real-time optimization (< 400ms)
- ✅ Apple Silicon optimization

---

## 🔗 Links

- **Portfolio** : [fabienpierret.github.io/projects/poker](https://fabienpierret.github.io)
- **Author** : [Fabien Pierret](https://github.com/fabienpierret)

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ for demonstrating computer vision and real-time optimization capabilities**

