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
The project has evolved into three distinct phases, blending physical earth sciences with advanced software engineering:

**Phase 1: Core Physical Sciences**
*   `Chapter_01/` - Climatic Variables & STAC Acquisition
*   `Chapter_02/` - Spectral Signature Analysis
*   `Chapter_03/` - Topography & Glacial Retreat
*   `Chapter_04/` - Ecological Niche & Climate Vulnerability
*   `Chapter_05/` - Zonal Statistics & Moisture Stress
*   `Chapter_06/` - Advanced Hydrometeorology
*   `Chapter_07/` - Radar & Multi-Sensor Review (SAR)

**Phase 2: Advanced Modeling & Geo-AI**
*   `Chapter_08/` - Data Fusion & The Cascade Effect (Machine Learning)
*   `Chapter_09/` - Multi-Task Deep Learning (PyTorch)
*   `Chapter_10/` - Agentic Orchestration (LangGraph)
*   `Chapter_11/` - Enterprise Spatial Databases (PostGIS)

**Phase 3: The Capstones**
*   `Chapter_12/` - Basic Capstone (Automated Site Analysis)
*   `Chapter_13_14/` - Advanced Capstone (FastAPI Microservice)

## 🚀 Getting Started
To replicate this project, please navigate to the respective Chapter folder and read its specific `README.md` file, starting with `Chapter_01`. Each chapter contains its own mathematical concepts, action steps, and required data inputs.

---

## 🗺️ GIS Interoperability (ArcGIS Pro & ENVI)
A core academic requirement of this curriculum is **Hybrid Dual-Track Interoperability**. Every Python script in this chapter has been engineered to automatically export **Geocoded TIFF (.tif)** and **Shapefile (.shp)** outputs into the \data/processed/\ directory.

Instead of just looking at matplotlib PNG graphs, students are encouraged to:
1. Run the automated Python pipeline.
2. Open **ArcGIS Pro** or **ENVI**.
3. Drag-and-drop the generated \.tif\ and \.shp\ files directly into the GUI.
4. Verify that the Python outputs align perfectly with the GUI software basemaps.
