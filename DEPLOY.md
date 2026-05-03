# Vercel Deployment Guide

This document explains how to configure and use automated Vercel deployments for the Lakeside Orchestration UI (Vite + React app).

---

## One-time Setup

### 1. Obtain `VERCEL_TOKEN`

1. Go to <https://vercel.com/account/tokens>.
2. Click **Create Token**, give it a descriptive name (e.g. `QRATUM-CI`), and copy the value.

### 2. Obtain `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID`

Run the following from the repository root (requires [Vercel CLI](https://vercel.com/docs/cli)):

```bash
npm install --global vercel
vercel link
```

Follow the prompts to link the project to your Vercel account. Afterwards, read the generated file:

```bash
cat .vercel/project.json
```

The file contains `orgId` (`VERCEL_ORG_ID`) and `projectId` (`VERCEL_PROJECT_ID`).

> **Note:** `.vercel/` is excluded from git (`.gitignore`). Never commit this directory.

---

## Adding Secrets to GitHub

Add the three secrets in the repository at:

**GitHub → Settings → Secrets and variables → Actions → New repository secret**

| Secret name | Value |
|---|---|
| `VERCEL_TOKEN` | Token from <https://vercel.com/account/tokens> |
| `VERCEL_ORG_ID` | `orgId` from `.vercel/project.json` |
| `VERCEL_PROJECT_ID` | `projectId` from `.vercel/project.json` |

Until these secrets are configured, the workflow will fail with an authentication error — this is expected on the first run.

---

## How Automated Deployment Works

The workflow at `.github/workflows/vercel-deploy.yml` defines two jobs:

| Job | Trigger | Environment |
|---|---|---|
| `deploy-preview` | Pull request targeting `MoSaaS` | Preview |
| `deploy-production` | Push to `MoSaaS` | Production |

### Preview deployments

When you open or update a pull request targeting `MoSaaS`, the workflow:

1. Installs Node 20 and runs `npm ci`.
2. Installs the Vercel CLI and pulls the preview environment configuration.
3. Builds the project with `vercel build`.
4. Deploys to a unique preview URL with `vercel deploy --prebuilt`.
5. Posts the preview URL as a sticky comment on the pull request.

### Production deployments

When a push lands on `MoSaaS` (e.g. after merging a PR), the workflow:

1. Installs Node 20 and runs `npm ci`.
2. Installs the Vercel CLI and pulls the production environment configuration.
3. Builds the project with `vercel build --prod`.
4. Deploys to production with `vercel deploy --prebuilt --prod`.

---

## Manual Fallback (Import via Vercel Dashboard)

If you prefer to deploy without the GitHub Actions workflow:

1. Go to <https://vercel.com/new>.
2. Click **Import Git Repository** and select `robertringler/QRATUM`.
3. Set the **Branch** to `MoSaaS`.
4. Vercel will auto-detect the framework as **Vite** via `vercel.json`.
5. Confirm the settings:
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
6. Click **Deploy**.

---

## Triggering a Redeploy

- **Production:** Push any commit to the `MoSaaS` branch (e.g. `git push origin MoSaaS`).
- **Preview:** Open or update a pull request targeting `MoSaaS`.
- **Manual:** Use the Vercel dashboard → your project → **Redeploy**.
