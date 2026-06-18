# P-CORE — Project Page

Project page for **P-CORE: Self-Supervised Surface Consistency for Point-Based Neural Editing** (ECCV 2026).

🔗 https://zvict.github.io/p-core/

> This is the **`gh-pages`** branch — it holds only the project website. The P-CORE **code** lives on the **`main`** branch of the same `zvict/p-core` repo. The two branches have independent histories (standard for a GitHub Pages branch).

- **Paper:** `static/pdfs/pcore_eccv2026.pdf`
- **Code:** coming soon (on `main`)
- **arXiv:** coming soon

Static site (no build step) based on the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).
Edit `index.html` to update content. Run `python3 tools/check_site.py` as a sanity gate.

## Local preview
```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Deploy (GitHub Pages — `gh-pages` branch)
1. Push this website branch:
   ```bash
   git push -u origin gh-pages
   ```
2. On GitHub: **Settings → Pages → Source: Deploy from a branch → `gh-pages` / `/ (root)` → Save**.
3. Push your code to `main` of the same repo separately.

To update the site later, commit to `gh-pages` and `git push origin gh-pages`.
