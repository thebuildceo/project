# GitHub Upload and Deployment Guide

This guide will help you upload the project to GitHub and deploy it on Streamlit Community Cloud.

**Repository**: https://github.com/thebuildceo/project

## Step 1: Initialize Git Repository

```bash
cd d:/WORK/customer_churn_prediction
git init
```

## Step 2: Add All Files to Git

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Customer Churn Prediction Dashboard"
```

## Step 4: Link Local Repository to GitHub

```bash
git remote add origin https://github.com/thebuildceo/project.git
git branch -M main
git push -u origin main
```

## Step 5: Deploy to Streamlit Community Cloud

### Option A: Automatic Deployment from GitHub

1. Go to https://share.streamlit.io
2. Sign up/login with your GitHub account
3. Click "New app"
4. Select your repository: `thebuildceo/project`
5. Select branch: `main`
6. Main file path: `app.py`
7. Click "Deploy"

### Option B: Manual Deployment

If automatic deployment doesn't work, you can manually deploy:

1. Make sure your repository is public on GitHub
2. In Streamlit Cloud, click "New app"
3. Paste your repository URL
4. Follow the same steps as Option A

## Step 7: Verify Deployment

1. Wait for the deployment to complete (usually 2-5 minutes)
2. Streamlit Cloud will provide a URL like: `https://your-app-name.streamlit.app`
3. Test all pages:
   - Home page
   - Data Analysis
   - Predict Churn
   - Model Insights

## Important Notes

### Data and Model Files

The `data/` and `models/` directories are included in the repository because:
- The dataset is small (~300KB)
- The trained model is small (~1MB)
- Streamlit Cloud needs these files to run the application
- No external database or API is required

### Requirements.txt

The `requirements.txt` file is in the root directory and will be automatically detected by Streamlit Cloud for dependency installation.

### Virtual Environment

The virtual environment is NOT included in the repository (it's in `.gitignore`). Streamlit Cloud creates its own environment.

### Screenshots

Screenshots are optional for deployment but recommended for the README. You can add them later by:
1. Running the app locally
2. Taking screenshots of each page
3. Adding them to the `screenshots/` directory
4. Committing and pushing to GitHub

## Troubleshooting

### Deployment Fails

- Check that the repository is public
- Verify `requirements.txt` is in the root directory
- Ensure `app.py` is in the root directory
- Check Streamlit Cloud logs for specific errors

### Model Not Found Error

- Make sure `models/churn_model.pkl` and `models/data_processor.pkl` exist
- Run `python train_model.py` locally to generate them
- Commit and push the models directory

### Dataset Not Found Error

- Make sure `data/telco_churn.csv` exists
- Run `python download_data.py` locally to download it
- Commit and push the data directory

### Import Errors

- Verify all dependencies are in `requirements.txt`
- Check that the `src/` directory has `__init__.py`
- Ensure import paths are correct in all Python files

## Success Criteria

Your deployment is successful when:
- ✅ The app loads without errors
- ✅ All pages are accessible
- ✅ Predictions work correctly
- ✅ Visualizations display properly
- ✅ No API keys or external services are required

## Next Steps

After successful deployment:
1. Share the Streamlit Cloud URL with your professor
2. Add screenshots to the README for better presentation
3. Prepare for viva by understanding:
   - The ML pipeline (data preprocessing, model training, inference)
   - Feature importance and model interpretation
   - Why Random Forest was chosen
   - The business value of churn prediction
