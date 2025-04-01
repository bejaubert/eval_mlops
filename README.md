
# 🏥 API - Prédiction de Réadmission Hospitalière

Ce projet vise à prédire la probabilité de réadmission d’un patient à l’hôpital à l’aide d’un modèle XGBoost déployé localement via une API FastAPI.  
Le modèle est entraîné avec MLflow et enregistré dans son Model Registry.  
Les justifications des choix d’hyperparamètres, de métriques, et de conception sont documentées dans le dossier `explication`.

---

## 📦 Installation des dépendances

Avant de lancer l’API, installez les bibliothèques nécessaires :

```bash
pip install -r requirements.txt
```

> 📌 Ce fichier contient les dépendances du projet (FastAPI, MLflow, pandas, etc.) nécessaires au bon fonctionnement du modèle et de l’API.

---

## 🚀 Lancer l’API (sur Onyxia ou localement)

Dans un terminal à la racine du projet :

```bash
uvicorn main:app --host 0.0.0.0 --port 8050
```

> ⚠️ Le paramètre `--host 0.0.0.0` est requis pour que l’API soit accessible dans un environnement Onyxia. Le port 8050 est recommandé car compatible avec le proxy.

---

## 🌐 Accès à la documentation interactive

Ouvrir dans le navigateur :

```
https://<votre-session-onvx>.lab.sspcloud.fr/proxy/8050/docs
```

Par exemple :
```
https://user-benjamin389-901468-0.user.lab.sspcloud.fr/proxy/8050/docs
```

---

## 🧪 Utilisation de l’API

### 🔹 Requête POST `/predict`

**Exemple de corps JSON à envoyer** :

```json
{
  "chol": 8.5,
  "crp": 13.2,
  "phos": 7.1
}
```

**Réponse attendue** :

```json
{
  "prediction": 1,
  "interpretation": "Réadmission probable"
}
```

> `1` = réadmission probable  
> `0` = pas de réadmission probable

---

## ✅ Résumé

- Modèle XGBoost logué et servi avec MLflow
- Déploiement local via FastAPI
- Swagger UI disponible pour tester les prédictions
- Justifications disponibles dans le dossier `explication`

