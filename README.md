# Détection de botnet KGB sur flux NetFlow

## Contexte et objectifs
Ce projet propose une solution de détection de botnets sur des flux réseau NetFlow, fondée sur l’algorithme **KGB** et ses variantes optimisées.  
À partir du dataset **CTU‑13** (scénario 10), il calcule des métriques d’entropie et applique une analyse en composantes principales (PCA) afin d’isoler et de scorer les comportements suspects.  
Des modules dédiés permettent l’exécution de la détection ainsi que l’optimisation automatique des hyperparamètres (PCA, seuils) via un algorithme génétique.

## Objectifs
- Prétraiter et agréger des captures de trafic réseau (CTU‑13, scénario 10)  
- Calculer des mesures d’entropie et réaliser une PCA  
- Détecter les sources suspectes (IP) via deux scores (composantes majeures & mineures)  
- Évaluer la performance (TP, FP, FN, TN) sur un cas réel de botnet **DDoS UDP**

## Contenu du dépôt
| Fichier | Rôle |
|---------|------|
| `KGB_Entropy_Based_HP_Optimized.py` | Implémentation optimisée de l’algorithme KGB (PCA, seuils, évaluation) |
| `AI_Genetics_Algo_K_Means_Based_HP_Optimized.py` | Optimisation génétique (GA) des hyperparamètres KGB |
| `Netflow_Intrusion_Detection_Project.pdf` | Présentation du contexte, des objectifs et du dataset |
| `Rapport_KGB.pdf` / `Soutenance_KGB.pdf` | Rapport détaillé et support de soutenance |
| `KGB_article.pdf` | Article de Pevný *et al.* (2012) – base théorique du KGB |
| `An_empirical_comparison_of_botnet_detection.pdf` | Comparatif méthodologique des techniques de détection |

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-organisation/kgb-netflow-detection.git
cd kgb-netflow-detection
```

### 2. Créer un environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate   # sous Windows : venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prétraitement des données NetFlow
```python
from KGB_Entropy_Based_HP_Optimized import EnhancedKGB

kgb = EnhancedKGB(sample_size=1_000_000, verbose=True)
df = kgb.load_and_preprocess_data("path/to/ctu10.csv")
X = kgb.compute_entropies()
```

### 2. Détection d’anomalies
```python
df_pred = kgb.detect_anomalies(
    n_major_components = 16,
    threshold_PCA     = 0.99257,
    threshold_major   = 0.99968,
    threshold_minor   = 0.98109
)

tn, fp, fn, tp = kgb.compute_confusion_matrix()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
```

### 3. Optimisation des hyperparamètres (optionnel)
```bash
python AI_Genetics_Algo_K_Means_Based_HP_Optimized.py     --file_path path/to/ctu10.csv     --generations 50     --population_size 20
```

## Résultats
- Les métriques de performance (FPR, TPR, précision, rappel, F‑measure) sont affichées en console et sauvegardées dans `kgb_results/<run_id>/`.
- L’optimiseur génétique écrit ses résultats dans `genetic_optimizer_results/<run_id>/`.

## Références
- Pevný T., Rehák M., Grill M. (2012). *Identifying suspicious users in corporate networks*. **Computers & Security**.  
- García S., Pechoucek M., Grill M. (2014). *An empirical comparison of botnet detection methods*.  
- **CTU‑13 dataset** (scénario 10) — CTU University, 2011.

---

> *Licence : MIT — voir `LICENSE` pour plus de détails.*
