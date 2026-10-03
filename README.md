# 🎯 ArXiv Semantic Radar (WebAssembly Edition)

A 100% client-side, zero-backend semantic search engine for the daily arXiv feed.

This project solves the noise problem in academic research without relying on paid cloud APIs or heavy Python backends. It downloads a lightweight Machine Learning model (`all-MiniLM-L6-v2`) directly into the browser cache and executes vector embeddings and Cosine Similarity locally using **Transformers.js** and WebAssembly.

## Architecture & Tech Stack
- **Frontend:** Vanilla HTML/JS (Zero build steps, ultra-lightweight).
- **AI Engine:** `Transformers.js` running purely in the browser.
- **Data Source:** arXiv Atom XML API (routed via CORS proxy).
- **Deployment:** GitHub Pages via GitHub Actions.
- **Privacy:** 100% local inference. Your research queries never leave your browser.

## Try It Live
🚀 **[View the Live Application Here](https://tnm-yes-anyidea.github.io/preprint-digger/)** *(Update this link once deployed!)*

## Local Development
Because this is entirely client-side, you don't need `pip` or Node modules. Simply clone the repository and open `index.html` in your browser, or run a local server:

```bash
git clone [https://github.com/tnm-yes-anyidea/preprint-digger.git](https://github.com/tnm-yes-anyidea/preprint-digger.git)
cd arxiv-portfolio-project
python3 -m http.server 8080
```
Then navigate to http://localhost:8080

### 3. The Final Step (Settings Toggle)
Before you push these files, you need to tell GitHub to use your new YAML file instead of the legacy deployment method:

1. Go to your repository on GitHub.com.
2. Click **Settings** > **Pages** (in the left sidebar).
3. Under the **Build and deployment** section, look for the **Source** dropdown.
4. Change it from "Deploy from a branch" to **"GitHub Actions"**.

Now, simply run `git add .`, commit, and run `git push origin pages`. GitHub Actions will automatically read the `.yml` file, spin up a runner, and your site will be live on the internet in under two minutes.
