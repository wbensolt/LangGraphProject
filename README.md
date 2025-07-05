# 🧠 Twitter Post Generator & Evaluator with LangGraph & LangChain

Ce projet combine **LangGraph**, **LangChain** et **des LLMs comme Gemini et LLaMA** pour créer un système interactif capable de :

- 📝 Générer des tweets techniques ou viraux à partir d'une requête utilisateur.
- 🤖 Réfléchir sur la qualité du tweet généré et proposer des améliorations.
- 🔁 Boucler automatiquement entre génération et critique jusqu’à obtenir un meilleur résultat.

---

## 📁 Structure du projet

### 1. `chains.py`

Contient deux chaînes (`generation_chain` et `reflection_chain`) construites avec :
- Des `ChatPromptTemplate`
- Un LLM (LLaMA via Ollama ou Gemini)

Ces chaînes sont utilisées pour :
- **Générer** un tweet
- **Réfléchir** et **critiquer** le tweet généré

---

### 2. `agent_with_tools.py`

Un exemple d’**agent LangChain avec outils** :
- 🔍 `TavilySearchResults` : recherche web
- ⏰ `get_system_time()` : retourne l'heure système
- Utilise Gemini pour répondre à des questions complexes

---

### 3. `main_graph.py` (ou ton script principal)

Met en place un **flux LangGraph** :
1. Démarre avec une demande utilisateur (`HumanMessage`)
2. Enchaîne :
   - **Génération (`generate`)**
   - **Réflexion (`reflect`)**
   - Répète jusqu'à 3 itérations

Affiche :
- Le graphe au format ASCII et Mermaid
- L'historique final des messages (prompt / tweet / critique / tweet révisé, etc.)

---

## 🔧 Pré-requis

- Python 3.10+
- Un environnement virtuel : `venv` ou `conda`
- Un fichier `.env` avec ta clé Gemini si tu l’utilises :

```env
GOOGLE_API_KEY=ta_clé_API_google_genai
