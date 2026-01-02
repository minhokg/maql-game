# Multi Agent Q Learning (MAQL) in Repeated Games

---

This repository explores the behavior of **Multi-Agent Q-Learning (MAQL)** in repeated games and examines how incorporating **Learning with Opponent-Learning Awareness (LOLA)** affects cooperation and convergence.

The project compares:
- **Naive Multi-Agent Q-Learning**, where agents learn independently, and  
- **LOLA-based Multi-Agent Q-Learning**, where agents explicitly account for the learning dynamics of their opponents.


---
## Setup

* run "pip install -r requirements.txt" in the terminal
* run "pre-commit install" in the terminal

## Repository Structure
```repo
├── functions/
│ └── helper/
│ └── models/
│ └── utils/
├── demo.ipynb
├── plots/ # Generated plots and figures
├── requirements.txt # Python dependencies
└── README.md
```
