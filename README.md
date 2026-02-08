# 🎓 Fuzzy Expert System for Student Academic Performance Evaluation

**Course Project: Fuzzy Sets and Applications**

A comprehensive fuzzy expert system implementing the compositional rule of inference for evaluating student academic performance. Features step-by-step visualization of the fuzzy inference process.

---

## 📁 Project Structure

```
fuzzy/
├── app.py                          # 🌐 Main Streamlit application (step-by-step demo)
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 This file
│
├── src/
│   └── fuzzy_expert_system.py      # 🔧 Core fuzzy system implementation
│
├── data/
│   └── sample_students.csv         # 📊 Sample dataset (120 student records)
│
├── docs/
│   └── DOCUMENTATION.md            # 📚 Complete academic documentation
│
└── outputs/
    ├── membership_functions.png    # 📈 Membership functions visualization
    ├── evaluation_excellent.png    # 📊 Excellent student evaluation
    ├── evaluation_struggling.png   # 📊 Struggling student evaluation
    └── response_surface.png        # 🗺️ 3D response surface
```

---

## 🚀 Quick Start

### Installation

```bash
cd /Users/papazov/Desktop/fuzzy

# Install dependencies
pip install -r requirements.txt
```

### Run the Interactive Application

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Run the Core System (CLI)

```bash
python src/fuzzy_expert_system.py
```

---

## 🌐 Streamlit Application Features

### Tab 1: 🔬 Evaluation Wizard

Interactive step-by-step demonstration of the complete fuzzy inference process:

| Step | Name | Description |
|------|------|-------------|
| 1 | **Load Data** | Upload CSV or use sample dataset (120 students) |
| 2 | **Review Rules** | Examine the 12 expert fuzzy rules |
| 3 | **Process Data** | Evaluate all students with progress tracking |
| 4 | **Select Student** | Choose a student for detailed analysis |
| 5 | **Fuzzification** | Visualize membership degree calculations |
| 6 | **Rule Evaluation** | See rule activations and firing strengths |
| 7 | **Final Result** | View performance score with gauge visualization |

### Tab 2: 📖 About the System

- Complete system overview and methodology
- Membership function visualizations for all variables
- Full rule base with explanations
- Inference method documentation (Mamdani)

---

## 📊 Sample Data Format

The `data/sample_students.csv` contains 120 student records with the following columns:

| Column | Description | Range |
|--------|-------------|-------|
| Student_ID | Unique identifier | STU-001 to STU-120 |
| Student_Name | Full name | Text |
| GPA | Grade Point Average | 2 - 6 (Bulgarian scale) |
| Absences | Classes missed | 0 - 30 |
| Participation | Class engagement % | 0 - 100 |
| Test_Score | Examination score % | 0 - 100 |
| Department | Academic department | Text |
| Year | Study year | 1 - 4 |

---

## 🔧 System Overview

### Input Variables

| Variable | Range | Linguistic Terms |
|----------|-------|------------------|
| GPA | 2-6 | Low, Medium, High |
| Absences | 0-30 | Few, Moderate, Many |
| Participation | 0-100 | Passive, Moderate, Active |
| Test Score | 0-100 | Poor, Average, Good, Excellent |

### Output Variable

| Variable | Range | Linguistic Terms | Method |
|----------|-------|------------------|--------|
| Performance | 0-100 | Low, Medium, High | Centroid |

### Rule Base (12 Expert Rules)

```
R1:  IF GPA=High ∧ Test=Excellent ∧ Part=Active ∧ Abs=Few → HIGH
R2:  IF GPA=High ∧ Test=Good → HIGH
R3:  IF GPA=High ∧ Test=Average → MEDIUM
R4:  IF GPA=Medium ∧ Test=Average ∧ Part=Moderate → MEDIUM
R5:  IF GPA=Medium ∧ Part=Active ∧ Abs=Few → MEDIUM
R6:  IF GPA=Low ∧ Test=Poor → LOW
R7:  IF Abs=Many ∧ Part=Passive → LOW
R8:  IF GPA=Low ∧ Part=Active ∧ Test=Average → MEDIUM
R9:  IF GPA=Medium ∧ Test=Excellent → HIGH
R10: IF Test=Good ∧ Abs=Few ∧ Part=Moderate → MEDIUM
R11: IF Test=Poor ∧ Part=Passive → LOW
R12: IF GPA=Medium ∧ Test=Good ∧ Abs=Moderate → MEDIUM
```

---

## 📐 Inference Method

| Component | Method |
|-----------|--------|
| Architecture | Mamdani Fuzzy Inference System |
| AND Operator | Minimum (min) |
| OR Operator | Maximum (max) |
| Implication | Mamdani (clipping) |
| Aggregation | Maximum |
| Defuzzification | Centroid (Center of Gravity) |

---

## 📚 Academic Documentation

Complete documentation is available in `docs/DOCUMENTATION.md` with:

1. Introduction
2. Theoretical Background of Fuzzy Sets
3. Problem Statement and Object Description
4. Design of the Fuzzy Expert System
5. Implementation and Applied Algorithms
6. Experimental Examples and Results
7. Analysis and Interpretation of Results
8. Conclusion
9. References

---

## 📦 Dependencies

```
numpy>=1.21.0
scikit-fuzzy>=0.4.2
matplotlib>=3.5.0
networkx>=2.6.0
streamlit>=1.28.0
plotly>=5.18.0
pandas>=1.5.0
```

---

## 📸 Screenshots

The application provides rich visualizations including:

- 📊 Interactive membership function plots
- 📈 Rule activation bar charts
- 🎯 Aggregated output fuzzy set visualization
- 📍 Centroid defuzzification animation
- 🥧 Performance distribution pie charts
- 📊 Histogram distributions

---

## 👨‍🎓 Course Information

**Course:** Fuzzy Sets and Applications

**Project Topic:** Design and Implementation of a Fuzzy Expert System for Evaluating Student Academic Performance Based on Precise and Expert Knowledge, Using the Compositional Rule of Inference

**Date:** February 2026

---

## 📝 License

Academic Project - For Educational Purposes
