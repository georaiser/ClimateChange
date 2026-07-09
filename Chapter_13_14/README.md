# Chapters 13 & 14: Advanced Geo-AI & Web API Capstone

## 🎯 Academic Objective
Welcome to the Advanced Capstone! 

Once you have mastered the physical sciences and automated vector-raster modeling in Chapters 1-12, the next step for an enterprise Geospatial Data Scientist is to deploy those models to the cloud.

In this upcoming advanced module, you will learn to:
1. **Build a REST API**: We will use **FastAPI** to wrap the `capstone_pipeline.py` script.
2. **Deploy as a Microservice**: Instead of running the script in a local terminal, a user will be able to send an HTTP POST request containing a Bounding Box (`{"bbox": [-72.8, -51.8, -72.4, -51.6]}`) to your server.
3. **Automated Response**: The server will instantly run the STAC extraction, calculate the NDVI and buffers, and return the `site_analysis_report.md` statistics as a JSON payload, alongside the Geocoded TIFFs.

*Note: This advanced software engineering chapter is currently a placeholder and will be developed as part of the Phase 2 (Chapters 7-11) architecture rollout.*
