# Modern Methods with TLS and UAV Group 2

---

## 🚀 How to Contribute to This Project

To keep our forestry data and scripts organized, please follow this standard Git workflow every time you work on the project.

### 1. Initial Setup (First time only)

If you haven't downloaded the project to your computer yet, run:

```powershell
git clone https://github.com/your-username/Modern-Methods-Group-2.git

```

### 2. The Daily Workflow

Follow these steps **every time** you want to make a change:

#### **Step A: Sync with the Group**

Before you start typing any code or moving any TLS files, make sure you have the latest version from everyone else.

```powershell
git pull origin main

```

#### **Step B: Make Your Changes**

Open your files, run your processing scripts, or add your UAV data analysis. Save your files as usual.

#### **Step C: Stage the Changes**

Tell Git which files you want to prepare for the upload. The `.` means "add everything."

```powershell
git add .

```

#### **Step D: Commit the Work**

Create a "snapshot" of your work with a clear message describing what you did (e.g., "Added noise filter to LiDAR script").

```powershell
git commit -m "Brief description of what you changed"

```

#### **Step E: Push to the Cloud**

Upload your snapshot so the rest of the group can see it.

```powershell
git push origin main

```

---

### ⚠️ Important Rules for This Project

* **Pull Before Push:** Always run `git pull` before you start working to avoid merge conflicts.
* **Small Commits:** It is better to push 5 small changes than one giant update. It makes it easier to fix if something breaks!
* **Large Data:** Do **not** commit raw `.las` or `.tif` files larger than 50MB. Use the shared cloud drive for raw data and Git for the processing scripts.