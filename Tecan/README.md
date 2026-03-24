# 🧪 Tecan Reader

A modular Python tool to read, transform, and aggregate data from Spark Multimode Microplate Reader Excel files. It currently supports fluorescence and absorption (general or two specific pigments) measurements, and outputs results in a structured format ready for analysis.

---

## 📚 Table of Contents

- [Version Log](#-version-log)
- [Manual](#-manual)
  - [Introduction](#introduction)
  - [Folder & File Structure](#folder--file-structure)
  - [How to Use](#how-to-use)
  - [Method Versions](#method-versions)
- [Planned Features](#-planned-features)
- [Credits](#-credits)

---

## 🕓 Version Log

### Version 0.1

- Initial version supporting:
  - Fluorescence extraction
  - Absorption extraction (general or two specific pigments)
  - Structured result writing to Excel
- Modular method handling
- Automatic updating of result file
- Designed for easy extensibility

### Version 0.2

- Fluorescense: Now useable if measured only once
- Minor bug fixes
- Results are now sorted before saved in the results file

---

## 📖 Manual

### 🧾 Introduction

This project automates the processing of output from Tecan plate readers. The script reads multiple Excel files from an input directory, extracts and transforms relevant data based on the measurement method, and writes the results into a clean, central Excel file.

Since there are several measuring types, possible dilutions and replicates, several individual information need to be written into the excel files. To understand how to add those information continue in the section 'How to use'.

The tool is modular and extensible, so new measurement types or transformation logics can be added with minimal changes to the core logic.

---

### 🗂️ Folder & File Structure

Tecan/
├── main.py
├── Coordinator.py
├── README.md
├── Evaluation Folder/
│   ├── Example 1.xlsx
│   ├── Example 2.xlsx
│   ├── Input/
│   └── Output/
├── Extraction/
│   ├── AbsorptionCoordinator.py
│   ├── ExtractFluorescence.py
│   ├── Reader/
│   │   └── BlockReading.py
│   └── Transformation/
│       ├── TransformationBlue.py
│       └── TransformationRed.py
└── ReadAndWrite/
    ├── Name_Reader.py
    └── Write.py



---

### 🛠️ How to Use

1. **Prepare the input Excel files**

   Each input file must contain three specific metadata rows at the top. These rows must each begin with the word "Ex" and provide information about the sample names, the measurement methods, and the dilution factors. An Example file can be found in the Evaluation Folder.

   - The first row should list the sample names from left to right. Include all blanks and make sure every column contains an entry. Do not leave empty cells.

     Example:
     ```
     [Ex ][Blank][Sample1][Sample2][Sample3]
     ```

   - The second row should indicate the measurement method used for each sample. Use the following abbreviations:
     - `f` for fluorescence
     - `b` for absorption (blue)
     - `r` for absorption (red)

     You can find more method codes in the "Method Versions" section if needed.

     Example:
     ```
     [Ex][Blank][f][f][b]
     ```

   - The third row should define the dilution factor for each sample. Use `1` for undiluted samples, `10` for a 1:10 dilution, and so on. Even blanks should have a number, typically `1`.

     Example:
     ```
     [Ex][1][1][2][10]
     ```

    - All rows should in the end have the same length and have, because of the "Ex" cell, have one more cell incluced than the amount of cells filled

2. **Place the input files into the correct folder**

   All input Excel files should be saved in the following directory:
   ```
   Evaluation Folder/Input/
   ```

3. **Choose the appropriate method version**

   Open the file `main.py` and specify which method version should be applied. Each method version corresponds to a different transformation logic. Further details can be found in the "Method Versions" section.

4. **Run the script**

   You can run the script either from a terminal or through an IDE like Spyder. Use the following command:
   ```
   python main.py
   ```

5. **Check the output**

   After the script has run, the processed results will be saved in:
   ```
   Evaluation Folder/Output/
   ```
   The output file contains the extracted data along with calculated averages, standard deviations, and other relevant values.

6. **Add new files later if needed**

   If you add more input files at a later time, simply place them into the input folder. The script will automatically detect and process them, updating the output file accordingly. if the same sample if measured again, the new data overwrites the old, if the sample is not detected in the file, it will be added to it.





### 🔬 Method Versions

Currently, the code can process the data from fluorescence measurements and absorbance. Which methods are used is described by a Version-variable, which can be changed in `main.py`. There are only two Versions available at the moment, which are explain further down in this section. 
However, there are limitations for now. Since the code is build modular, further or other methods can be added easily. The methods included are:

- `f`: **Fluorescence** — handled by `ExtractFluorescence.py`
- `b`: **Blue Absorption** — handled by `AbsorptionCoordinator.py` and `TransformationBlue.py`
- `r`: **Red Absorption** — handled by `AbsorptionCoordinator.py` and `TransformationRed.py`

These are coordinated in `Coordinator.py`, and new methods can be added by following the structure of the existing ones. This is how those methods work:

Fluorescence:
The Blank for this method needs to be called "Blank". The value at that position will be subtracted from the values labled with the method "f". This method will search for the fluorescence measured at a wavelength of 544 nm. The dilution is implemented before the blank is subtracted. The average of all samples with the same name and the standard deviation will be saved in the output.

Blue Absorption:
This method is used to analyse the data for the extraction of actinohordin. The Blank for this method needs to be called "BlankB". The value at that position will be subtracted from the values labled with the method "b". The dilution is implemented before the blank is subtracted. Depending on the Version, either the values are furthe used to calculate the concentration (in mol/L) or left "raw". The average of all samples with the same name and the standard deviation will be saved in the output.

Red Absorption:
This method is used to analyse the data for the extraction of undecylpriogisin. The Blank for this method needs to be called "BlankR". The value at that position will be subtracted from the values labled with the method "r". The dilution is implemented before the blank is subtracted. Depending on the Version, either the values are furthe used to calculate the concentration (in mol/L) or left "raw". The average of all samples with the same name and the standard deviation will be saved in the output.

#### Version

1. LS
    This Version of the methods will search for "f", "b" and "r". This code does also work if one of those is missing. Regarding "b" and "r" the values are in this Version transformed into the resulting concentration of the pigments.

1. Raw
    This Version of the methods will search for "f", "b" and "r". This code does also work if one of those is missing. Regarding "b" and "r" the values are in this Version are not transformed into the resulting concentration of the pigments. Thus they are kept raw and the average and the standard deviation is saved.

---

## 🧭 Planned Features

- Support for additional measurement types
    - Easy set up for all Tecan users
- GUI for non-technical users
- Optional configuration file for flexible setups
- Visualization tools (plots, trends)
- Adding a method for the fluorescense to choose a value at another wavelength
    - specific values
    - highest values for the "b" method

---

## 👩‍💻 Credits

Me.
Though some credits may go to Monster Energy for keeping me awake

---



