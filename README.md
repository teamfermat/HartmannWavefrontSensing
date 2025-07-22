# Hartmann Wavefront Sensor Simulation for Gravitational Wave Detectors

## Project Overview

This repository makes data for the publication ... public.
This repository contains the code and simulation environment for the paper:

> *A Simulation Framework for Configuring Hartmann Wavefront Sensors in Gravitational Wave Detectors*.

The goal of this project is to evaluate and optimize the performance of Hartmann wavefront sensors (WFS) used for precise optical measurements in gravitational wave detectors, such as the Einstein Telescope.

Thermal deformation in the optics poses challenges for accurate measurement, and individually tailored configurations of WFS are required. This simulation provides a method to:

- Analyze and compare different WFS configurations.
- Quantify sensor performance via a defined figure of merit.
- Explore the effects of core parameters: number of apertures, aperture diameter, and Hartmann plate-to-camera distance.
- Define a common measurement range to enable fair comparisons across setups.

---

## Repository Structure
📁 WFS_20240228
└── Main simulation program and core functions

📁 WFS_Import
└── Subroutines and utility functions

📁 Simulation data - Curves for sensor performance with noise 1
📁 Simulation data 2 - Curves for sensor performance with noise 10

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://gitlab.com/your_username/your_repo_name.git
   cd your_repo_name
2. Open the main simulation file inside WFS_20240228 to begin your analysis.
3. Or use the data folders to visualize performance under different noise conditions.




