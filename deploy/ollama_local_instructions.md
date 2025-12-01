Ollama local — Instructions pour utiliser `llama3.2` avec le backend FitBox

Contexte
- Vous avez un serveur Ollama local et le modèle `llama3.2` disponible (ex: `llama3.2:latest`).
- Le backend `backend/backend_api.py` supporte désormais l'envoi des prompts à Ollama local ou Cloud.

Problème fréquent: "bind: address already in use"
- Message vu: `Error: listen tcp 127.0.0.1:11434: bind: address already in use`
- Cela signifie qu'une autre instance d'Ollama est déjà en écoute sur le port `11434`.

Vérifications et actions
1) Voir les modèles disponibles (vous avez déjà fait):
```bash
ollama list
# Ex: affichera llama3.2:latest, llama3.2:3b
```

2) Vérifier si un serveur Ollama tourne et sa PID:
```bash
# voir processus écoutant sur 11434
sudo ss -ltnp | grep 11434 || true
# ou (si ss absent)
netstat -ltnp | grep 11434 || true
```
- Si un processus est listé, notez son PID (colonne PID/Program name).

3) Si vous voulez redémarrer Ollama (arrêter l'instance existante puis relancer):
```bash
# tuez le PID observé (remplacez <PID>)
kill <PID>
# ou plus fort
kill -9 <PID>

# relancez Ollama (si installé globalement):
ollama serve
# si vous voulez forcer un port différent (ex 11435):
OLLAMA_PORT=11435 ollama serve
```

4) Configurer le backend FitBox pour utiliser le serveur local
- Exporter les variables d'environnement avant de lancer le backend (dans le même shell):
```bash
# utilisation du serveur local par défaut
export OLLAMA_LOCAL=1
export OLLAMA_LOCAL_URL='http://127.0.0.1:11434/api/generate'  # modifiez si vous aviez changé le port
export OLLAMA_MODEL_NAME='llama3.2:latest'                     # ou 'llama3.2:3b'

# lancer le backend
python3 backend/backend_api.py
```
- Si vous préférez indiquer directement une URL distante (Ollama Cloud), utilisez `OLLAMA_API_URL` et `OLLAMA_API_KEY`.

5) Tester l'endpoint local `/chat` (après démarrage du backend)
```bash
curl -sS http://localhost:5000/chat -X POST -H "Content-Type: application/json" \
  -d '{"user_data":{"age":30,"gender":"male","weight":70,"height":1.75},"message":"Donne un court programme d\"entraînement pour débutant."}'
```

Notes techniques
- Le backend enverra un JSON à l'endpoint Ollama. Pour un serveur local, pas besoin de clé d'API.
- Si vous utilisez un port non standard, exportez `OLLAMA_LOCAL_URL` en conséquence.

Nettoyage (optionnel)
- Si vous voulez supprimer les poids locaux du repo pour éviter la confusion (vous passez tout sur Ollama) :
```bash
rm -rf models/fitbox_model
```

Si quelque chose échoue
- Copiez/collez l'erreur complète retournée par le backend (stdout) et le contenu brut de la réponse HTTP d'Ollama (si présente). Je l'adapterai pour assurer le parsing correct de la réponse.

---
Faites-moi savoir si vous voulez que j'ajoute un endpoint `/test-ollama` pour renvoyer la réponse brute d'Ollama (utile pour debug).