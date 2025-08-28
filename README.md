# DOSCAR GUI Plotter

## Purpose

The **DOSCAR GUI Plotter** is a Python-based web application designed to visualize Density of States (DOS) data from VASP calculations. It supports both spin-polarized (ISPIN=2) and non-spin-polarized (ISPIN=1) calculations, as well as various orbital projection schemes (LORBIT settings). The application provides an intuitive interface for uploading data, customizing plots, and exporting high-quality PNG visualizations.

---

## Features

- **Spin-Polarized and Non-Spin-Polarized Support**:
  - Handles ISPIN=1 (non-spin-polarized) and ISPIN=2 (spin-polarized) calculations.
  
- **Orbital Projection Support**:
  - Supports LORBIT settings:
    - **LORBIT=11-14**: Individual orbitals are lm-resolved/decomposed (e.g., px, py, pz, etc.).
    - **LORBIT=0, 1, 2, 5, 10**: Orbitals are grouped (e.g., total p-orbital contributions).

- **Customizable Graphs**:
  - Total DOS is plotted as the sum of atomic contributions, similar to wxDragon.
  - The total DOS is **not** plotted as in the first block (column 2) of the DOSCAR file because VASP projects a finite set of PAW spheres, excluding interstitial or delocalized sites.

- **Legend Sorting**:
  - The legend is sorted by **Mendeleev numbers** for intuitive ordering of elements.

- **Input Requirements**:
  - The uploaded ZIP file must contain a folder with the following:
    - **POSCAR**: Contains structural information.
    - **DOSCAR**: Contains DOS data.
  - The folder being zipped must be named after the molecule/compound title (e.g., `Gd10RuCd3`), as the graph title is derived from the folder name.
  - The ZIP file itself can have any name.

---

## Hosting

This application is hosted on **Heroku** and can be accessed at the following link:

[https://doscar-gui-9a48cc5d4e0d.herokuapp.com](here!)

---

## DOSCAR File

The file **DOSCAR** contains the DOS and integrated DOS. The units are "number of states/unit cell". For dynamic simulations and relaxations, an averaged DOS and an averaged integrated DOS are written to the file. For a description of how the averaging is done, see sections 6.20 and 6.36 of the VASP documentation.

The first few lines of the DOSCAR file are made up of a header, followed by NDOS lines holding three data:

```
energy     DOS     integrated DOS
```

The density of states (DOS) $\bar{n}$ is determined as the difference of the integrated DOS between two points:

![DOS Equation](https://latex.codecogs.com/png.image?\dpi{150}\bar{n}(\epsilon_i)=\frac{N(\epsilon_i)-N(\epsilon_{i-1})}{\Delta\epsilon})

where $\Delta \epsilon$ is the energy difference between two grid points in the DOSCAR file, and $N(\epsilon_i)$ is the integrated DOS:

![Integrated DOS](https://latex.codecogs.com/png.image?\dpi{150}N(\epsilon_i)=\int_{-\infty}^{\epsilon_i}n(\epsilon)\,d\epsilon)

This method conserves the total number of electrons exactly.

For spin-polarized calculations, each line holds five data:

```
energy     DOS(up) DOS(dwn)  integrated DOS(up) integrated DOS(dwn)
```

If **RWIGS** or **LORBIT** (Wigner-Seitz radii) is set in the INCAR file, an lm- and site-projected DOS is calculated and written to the file DOSCAR. One set of data is written for each ion, with NDOS lines containing:

```
energy s-DOS p-DOS d-DOS
```

or, for spin-polarized cases:

```
energy s-DOS(up) s-DOS(down) p-DOS(up) p-DOS(dwn) d-DOS(up) d-DOS(dwn)
```

For non-collinear calculations, the total DOS has the following format:

```
energy     DOS(total)   integrated-DOS(total)
```

Information on individual spin components is available only for the site-projected density of states, which has the format:

```
energy s-DOS(total) s-DOS(mx) s-DOS(my) s-DOS(mz) p-DOS(total) p-DOS(mx) ...
```

In this case, the (site-projected) total density of states (total) and the (site-projected) energy-resolved magnetization density in the $x$ (mx), $y$ (my), and $z$ (mz) directions are available.

In all cases, the units of the l- and site-projected DOS are states/atom/energy.

### Notes:
- The site-projected DOS is not evaluated in the parallel version for the following cases:
  - **vasp.4.5, NPAR ≠ 1**: No site-projected DOS.
  - **vasp.4.6, NPAR ≠ 1, LORBIT=0-5**: No site-projected DOS.
- In **vasp.4.6**, the site-projected DOS can be evaluated for **LORBIT=10-12**, even if **NPAR ≠ 1** (contrary to previous releases).
- For relaxations, the DOSCAR is usually useless. To get an accurate DOS for the final configuration, copy **CONTCAR** to **POSCAR** and continue with one static calculation (**ISTART=1; NSW=0**).

For more details, refer to the following [VASP documentation](https://www.smcm.iqfr.csic.es/docs/vasp/node69.html).

---

## Useful Links

- [ISPIN](https://www.vasp.at/wiki/index.php/ISPIN)
- [DOSCAR](https://www.vasp.at/wiki/index.php/DOSCAR)
- [LORBIT](https://www.vasp.at/wiki/index.php/LORBIT)

---

## How It Works

### Parsing the DOSCAR File

- **First Block**:
  - The first block contains the total DOS and integrated DOS (IDOS). However, the total DOS is recalculated as the sum of atomic contributions to account for interstitial or delocalized sites.

- **Atomic Blocks**:
  - Each atomic block is parsed to extract orbital contributions. The number of columns in the atomic blocks determines the orbital resolution:
    - **4 Columns**: Grouped atomic contributions (e.g., s, p, d).
    - **5 Columns**: Individual orbitals resolved, without d orbital (e.g., s, px, py, pz, only).
    - **7 Columns**: Spin-polarized grouped contributions (e.g., s↑, s↓, p↑, p↓, etc.).
    - **10 Columns**: Individual orbitals resolved (e.g., px, py, pz, etc.).
    - **19 Columns**: Spin-polarized individual orbitals resolved (e.g., px↑, px↓, py↑, py↓, etc.).

### Legend Sorting

- The legend is sorted by **Mendeleev numbers**, ensuring that elements are displayed in a chemically intuitive order.

---

## Future Updates

For upcoming features and improvements, please refer to the project's GitHub Issues page:  
👉 [doscar-gui Issues](https://github.com/EmilJaffal/doscar-gui/issues)

Some planned updates include:

- Plotting **Integrated DOS (IDOS)**, including spin-resolved IDOS.  
- Supporting **individual atom contributions** (e.g., `Ge1`, `Ge2`) and their orbital-resolved plots.  
- Customizable **legend sorting**, such as by Mendeleev number, alphabetical order, or user-defined.

For details, suggestions, or to contribute, please visit the issues page above.

---

## Usage Instructions

### How to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/emiljaffal/doscar-gui
   cd /path/to/doscar-gui
   ```

2. **Install Dependencies**:
   - Ensure you have Python 3.10 or later installed.
   - Install the required Python packages:
     ```bash
     conda install --file requirements.txt
     pip install kaleido==0.2.1 dash==2.9.0
     ```

3. **Run the Application**:
   - Start the application:
     ```bash
     python app.py
     ```

4. **Access the Application**:
   - Open your browser and navigate to:
     ```
     http://127.0.0.1:8050/
     ```

5. **Upload Data**:
   - Upload a ZIP file containing a folder with `POSCAR` and `DOSCAR` files.
   - Ensure the folder name matches the molecule/compound title.

6. **Customize and Export**:
   - Adjust axis limits, legend position, and other settings.
   - Save the plot as a high-resolution PNG file.

---

## Example

### Input Folder Structure

```
Gd10RuCd3/
├── POSCAR
├── DOSCAR
```

### Output

- A PNG file of the DOS plot, similar to matplotlib, titled `Gd₁₀RuCd₃ DOS`.

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or bug reports.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
