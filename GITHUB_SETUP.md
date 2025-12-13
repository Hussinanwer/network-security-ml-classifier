# GitHub Setup Guide

Step-by-step guide to push this project to GitHub.

## Prerequisites

✅ Git installed on your computer
✅ GitHub account created
✅ Repository ready to push

## Step 1: Initialize Git (if not already done)

Open Command Prompt in the project directory and run:

```bash
git init
```

## Step 2: Configure Git (First Time Only)

Set your name and email (use your GitHub email):

```bash
git config --global user.name "Hussain Anwer"
git config --global user.email "your-email@example.com"
```

## Step 3: Create Repository on GitHub

1. Go to https://github.com/new
2. **Repository name**: `network-security-ml-classifier`
3. **Description**: "Machine learning system for network traffic classification with 100% accuracy"
4. **Visibility**: Choose Public or Private
5. **DO NOT** check "Initialize with README" (we already have one)
6. Click "Create repository"

## Step 4: Add Files to Git

Add all project files:

```bash
# Add all files
git add .

# Check what will be committed
git status
```

## Step 5: Create Initial Commit

```bash
git commit -m "Initial commit: Network Security ML Classifier

- Complete Streamlit dashboard for network traffic classification
- Three trained models (Random Forest 100%, Logistic Regression, SVM)
- Comprehensive preprocessing pipeline
- Windows batch files for easy setup
- Complete documentation and test suite
- Ready for team collaboration"
```

## Step 6: Connect to GitHub

Replace with your actual GitHub repository URL:

```bash
git remote add origin https://github.com/hussinanwer/network-security-ml-classifier.git
```

Verify the remote:
```bash
git remote -v
```

## Step 7: Push to GitHub

Push your code:

```bash
# For first push
git branch -M main
git push -u origin main
```

If you have authentication issues, see [Authentication Options](#authentication-options) below.

## Step 8: Verify on GitHub

1. Go to https://github.com/hussinanwer/network-security-ml-classifier
2. Verify all files are uploaded
3. Check that README displays correctly

## 🎉 Success!

Your repository is now live on GitHub!

---

## Authentication Options

### Option 1: Personal Access Token (Recommended)

1. **Generate token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Name: "Network Security ML Project"
   - Select scopes: `repo` (all checkboxes)
   - Click "Generate token"
   - **COPY THE TOKEN** (you won't see it again!)

2. **Use token when pushing:**
   ```bash
   git push -u origin main
   ```
   - Username: `hussinanwer`
   - Password: `[paste your token here]`

3. **Save credentials (optional):**
   ```bash
   git config --global credential.helper store
   ```

### Option 2: SSH Key

1. **Generate SSH key:**
   ```bash
   ssh-keygen -t ed25519 -C "your-email@example.com"
   ```
   Press Enter for default location

2. **Copy public key:**
   ```bash
   # Windows
   type %USERPROFILE%\.ssh\id_ed25519.pub

   # Linux/Mac
   cat ~/.ssh/id_ed25519.pub
   ```

3. **Add to GitHub:**
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Paste your public key
   - Click "Add SSH key"

4. **Change remote to SSH:**
   ```bash
   git remote set-url origin git@github.com:hussinanwer/network-security-ml-classifier.git
   ```

5. **Push:**
   ```bash
   git push -u origin main
   ```

---

## Common Commands for Team Collaboration

### Daily Workflow

```bash
# Get latest changes
git pull origin main

# Create new branch
git checkout -b feature/your-feature

# Make changes, then stage them
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push origin feature/your-feature

# Merge back to main (after PR approval)
git checkout main
git merge feature/your-feature
git push origin main
```

### Useful Commands

```bash
# Check status
git status

# View commit history
git log --oneline

# See what changed
git diff

# Undo uncommitted changes
git checkout -- .

# Update from remote
git fetch origin
git merge origin/main

# Clone repository (for teammates)
git clone https://github.com/hussinanwer/network-security-ml-classifier.git
```

---

## For Your Teammates

Share this with your team:

### 🚀 Quick Setup for Team Members

1. **Clone the repository**
   ```bash
   git clone https://github.com/hussinanwer/network-security-ml-classifier.git
   cd network-security-ml-classifier
   ```

2. **Install dependencies**
   - **Windows**: Double-click `INSTALL.bat`
   - **Linux/Mac**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate  # Windows: venv\Scripts\activate
     pip install -r requirements.txt
     python train.py
     ```

3. **Start working**
   ```bash
   streamlit run dashboard.py
   ```

### Making Changes

```bash
# Always pull latest changes first
git pull origin main

# Create your feature branch
git checkout -b feature/my-feature

# Make changes and commit
git add .
git commit -m "Add my feature"

# Push and create Pull Request
git push origin feature/my-feature
```

Then create a Pull Request on GitHub!

---

## Troubleshooting

### "fatal: not a git repository"
```bash
git init
```

### "remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/hussinanwer/network-security-ml-classifier.git
```

### "Permission denied"
- Use Personal Access Token instead of password
- Or set up SSH keys (see above)

### "refusing to merge unrelated histories"
```bash
git pull origin main --allow-unrelated-histories
```

### Large files causing issues
```bash
# Check file sizes
git ls-files | xargs ls -lh

# If models are too large, use Git LFS:
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## Git Best Practices for Team

1. **Always pull before push**
   ```bash
   git pull origin main
   ```

2. **Use descriptive commit messages**
   ```bash
   # Good
   git commit -m "Add confusion matrix visualization to dashboard"

   # Bad
   git commit -m "update"
   ```

3. **Commit often, push regularly**
   - Commit logical chunks of work
   - Push at end of day or when feature is done

4. **Never commit:**
   - `venv/` directory
   - `.env` file (use `.env.example` instead)
   - `__pycache__/` folders
   - Large temporary files

5. **Use branches for features**
   - `main` - stable code only
   - `feature/name` - new features
   - `bugfix/name` - bug fixes

---

## Quick Reference

```bash
# Setup (one time)
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/hussinanwer/network-security-ml-classifier.git
git push -u origin main

# Daily use
git pull origin main      # Get updates
git add .                 # Stage changes
git commit -m "message"   # Commit
git push origin main      # Push to GitHub

# Branching
git checkout -b feature/new-feature  # Create and switch
git push origin feature/new-feature  # Push branch
git checkout main                    # Switch back
git merge feature/new-feature        # Merge
```

---

**🎉 You're all set! Happy coding!**
