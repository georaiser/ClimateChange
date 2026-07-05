# GeoCascade: Comprehensive Climate, Glacial, and Watershed Analysis

Welcome to the ultimate academic and enterprise-grade geospatial pipeline. This project merges Climate Change analysis, Glaciology (Retreat Dynamics), Hydrology (Watershed Modeling), and advanced AI (Deep Learning, Agentic Orchestration, Web GIS).

## 📖 Project Philosophy
This project is designed with **absolute reproducibility** in mind. Every script, environment variable, and data source is explicitly documented so that any user, from a student to a senior data scientist, can replicate this entire 11-chapter pipeline step-by-step.

## 🛠️ System Prerequisites & Installation Guide
Before beginning Chapter 1, ensure your system is prepared. Because geospatial dependencies (like GDAL) are notoriously difficult to compile, we strictly use **Miniforge** (Mamba).

### 1. Python & Base Environment (Miniforge/Mamba)
We highly recommend installing [Miniforge](https://github.com/conda-forge/miniforge), which comes pre-configured with the `conda-forge` channel and the fast `mamba` package manager.
*   **Windows/Linux Installation:** Download the Miniforge installer for your OS and run it.
*   **Creating the Base Environment:** We use a modular approach, installing heavy packages only when needed for a specific chapter. First, create the lightweight base environment:
    ```bash
    mamba create -n geocascade_env python=3.12 -y
    ```
*   **Activation:**
    ```bash
    mamba activate geocascade_env
    ```
*   **Chapter-by-Chapter Installations:** At the start of each chapter, the specific required libraries will be installed (e.g., PyTorch in Chapter 2, PostGIS in Chapter 10). Check the `README.md` inside each chapter folder for the specific installation commands.

### 2. R & RStudio
For Chapter 4 (MaxEnt Modeling) and certain Chapter 1 analyses, R is required.
*   Download and install [R](https://cran.r-project.org/).
*   Download and install [RStudio Desktop](https://posit.co/download/rstudio-desktop/).

### 3. ArcGIS Pro
For Chapters 3, 4, and 5, ArcGIS Pro (with the Spatial Analyst extension) is used.
*   Ensure ArcGIS Pro is installed and licensed. The Python scripts utilizing `arcpy` must be run within the cloned ArcGIS Pro Python environment.

### 4. Docker (For Chapter 10 PostGIS)
Docker is required for the spatial database.
*   **Windows (Current Workspace for ArcGIS):** Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/).
*   **Linux (Debian/Ubuntu Server Deployments):** Run the following snippet to install the official Docker repo:
    ```bash
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo systemctl enable --now docker
    sudo usermod -aG docker "$USER"
    ```

### 5. AI & PyTorch Installation (For Chapters 2 & 9)
When we reach the AI chapters, you will need to install PyTorch with CUDA acceleration. The project uses CUDA 12.4. You can update your environment (or create it from scratch) using:
```bash
micromamba install -n geocascade_env pytorch torchvision torchaudio pytorch-cuda=12.4 opencv -c pytorch -c nvidia -c conda-forge --channel-priority flexible
```

*Note: You will also need free academic accounts for the Copernicus Data Space Ecosystem (CDSE) and NASA Earthdata to access the STAC APIs.*

## 📂 Project Structure
The project is divided into two phases: **Core Physical Sciences** (Ch 1-6) and **Advanced Geo-AI Architectures** (Ch 7-11).

*   `Chapter_01/` - Climatic Variables & Image Processing
*   `Chapter_02/` - Glacial Retreat, SAR Dynamics, and AI
*   `Chapter_03/` - Hydrology and Watershed Modeling
*   `Chapter_04/` - Vulnerability and Ecosystem Impacts
*   `Chapter_05/` - Cartography and Management Plans
*   `Chapter_06/` - The Linear Unified Pipeline
*   `Chapter_07/` - Agentic Orchestration (LangGraph)
*   `Chapter_08/` - The Cascade Effect Modeling
*   `Chapter_09/` - Multi-Task PyTorch Model
*   `Chapter_10/` - Spatial Database Integration (PostGIS)
*   `Chapter_11/` - Interactive Dashboard (Full-Stack GIS)

## 🚀 Getting Started
To replicate this project, please navigate to the respective Chapter folder and read its specific `README.md` file, starting with `Chapter_01`. Each chapter contains its own mathematical concepts, action steps, and required data inputs.
