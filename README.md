# Classification d'articles AG News : TF-IDF contre DistilBERT

Classer automatiquement des articles de presse dans 4 rubriques (World, Sports, Business, Sci/Tech), en comparant une baseline de machine learning classique au fine-tuning d'un Transformer pré-entraîné.

## Résultats

Les deux modèles sont entraînés sur le même échantillon de 20 000 articles et évalués sur les 7 600 articles du jeu de test officiel.

| Modèle | Accuracy | F1 macro | Paramètres | Temps d'entraînement |
|---|---|---|---|---|
| TF-IDF + régression logistique | 90,58 % | 90,54 % | ~50 000 (vocabulaire) | 18 s (CPU) |
| **DistilBERT fine-tuné (2 epochs)** | **92,32 %** | **92,31 %** | 67 M | 2,7 min (GPU) |

F1 par classe :

| Classe | Baseline | DistilBERT | Écart |
|---|---|---|---|
| World | 0,9084 | **0,9326** | +0,024 |
| Sports | 0,9608 | **0,9780** | +0,017 |
| Business | 0,8690 | **0,8834** | +0,014 |
| Sci/Tech | 0,8836 | **0,8985** | +0,015 |

**Lecture.** À données égales, DistilBERT gagne sur les quatre rubriques (+1,7 point d'accuracy). Business et Sci/Tech restent les plus difficiles pour les deux modèles, car elles partagent beaucoup de vocabulaire. Le gain est réel mais modeste pour un modèle 1 000 fois plus gros qui demande un GPU : la baseline reste pertinente quand le budget de calcul est limité.

## Données

[AG News](https://huggingface.co/datasets/fancyzhx/ag_news) : 120 000 articles d'entraînement, 4 classes équilibrées (25 % chacune). Le notebook télécharge le jeu depuis Hugging Face ; rien n'est stocké dans le dépôt.

- Longueur médiane : 50 tokens ; 99,2 % des textes font 128 tokens ou moins.
- Échantillon d'entraînement de 20 000 articles (`SUBSAMPLE_TRAIN`, modifiable), test sur les 7 600 articles officiels.

## Méthode

1. **Baseline** : TF-IDF sur uni- et bigrammes (50 000 termes max) puis régression logistique multinomiale.
2. **Transformer** : tokenisation WordPiece, `max_length = 128`, fine-tuning de `distilbert-base-uncased` (batch 32, 2 epochs, warmup 10 %, meilleur modèle retenu sur le F1).
3. **Évaluation** : accuracy, précision, rappel, F1 macro, comparaison par classe et inspection des erreurs.

## Lancer le projet

Recommandé : Google Colab avec un GPU (Exécution → Modifier le type d'exécution → GPU).

```bash
git clone https://github.com/DOWOU-Issa/ag-news-distilbert.git
cd ag-news-distilbert
pip install -r requirements.txt
jupyter notebook notebooks/ag_news_tfidf_vs_distilbert.ipynb
```

## Structure

```
ag-news-distilbert/
├── notebooks/ag_news_tfidf_vs_distilbert.ipynb   # pipeline complet
├── data/       # vide : le jeu est téléchargé à l'exécution
├── models/     # modèles sauvegardés (ignorés par Git)
├── results/    # figures exportées
└── requirements.txt
```

## Limites et pistes

- Entraîner les deux modèles sur les 120 000 articles pour vérifier que l'écart se maintient.
- Tester `bert-base-uncased` ou `roberta-base`.
- Mesurer le coût d'inférence (latence, mémoire) face à la baseline.

## Auteur

DOWOU Issa · [Portfolio](https://github.com/DOWOU-Issa) · [LinkedIn](https://www.linkedin.com/in/issa-dowou-4abbaa3a1)
