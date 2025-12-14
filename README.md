# AV Block Classification using Pushdown Automata

**Authors:** Salma Bellaou, Asma Khamri  
**University:** Mohammed VI Polytechnic University (UM6P)  
**Course:** Computational Theory - Final Project  
**Date:** December 2025

---

## 📖 About

This project uses **Pushdown Automata (PDA)** to automatically classify different types of heart rhythm problems (AV blocks) from ECG patterns. It's a practical application of computer science theory to medical diagnosis.

### What are AV Blocks?

AV blocks are heart conditions where electrical signals between the atria (upper chambers) and ventricles (lower chambers) are delayed or blocked. We classify four types:

- **First-Degree**: Constant delay, all beats pass through
- **Mobitz Type I**: Progressive delay until a beat drops
- **Mobitz Type II**: Sudden dropped beats
- **Third-Degree**: Complete block, independent rhythms

---

## 🎯 What We Built

✅ Working Python implementation of PDA classifier  
✅ Synthetic ECG data generator (100 test samples)  
✅ Experimental results with performance metrics  
✅ State diagram visualizations  
✅ Comparison with simpler models (FSA, Rule-Based)  

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| **Our PDA** | 60% |
| FSA Baseline | 60% |
| Rule-Based | 40% |

**Highlights:**
- 🎯 **100% accuracy** on Mobitz Type I (perfect!)
- 📈 **67% F1-score** on First-Degree blocks
- 📚 Demonstrates stack memory can capture temporal patterns

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/salma-bella/av-block-pda.git
cd av-block-pda

# Install dependencies
pip install -r requirements.txt
```

### Run the Code

```bash
# Run full experimental evaluation
python complete_implementation.py

# Generate visualizations
python create_visualizations.py
```

### Expected Output

The program will:
1. Generate 100 synthetic ECG sequences
2. Test our PDA classifier
3. Compare with baseline models
4. Display accuracy metrics
5. Save results to `experimental_results.json`

---

## 📁 Project Structure

```
av-block-pda/
├── README.md                          # You are here
├── complete_implementation.py         # Main code: PDA + experiments
├── pda_av_block.py                   # Detailed PDA with traces
├── create_visualizations.py           # Generates diagrams

```

---


## 🔬 How It Works

### 1. **Symbolic Encoding**

We convert ECG signals into symbols:
- `P` - P-wave (atrial beat)
- `R` - QRS complex (ventricular beat)
- `PRlong` - Long delay between P and R
- `PRincrease` - Delay is getting longer
- `PRnormal` - Normal delay

### 2. **PDA Processing**

The PDA uses a **stack** to remember patterns:
- Pushes symbols onto the stack
- Pops symbols when checking patterns
- Accepts when it detects a specific AV block pattern

### 3. **Classification**

Based on stack contents and patterns:
- **First-Degree**: Stack has 3+ `LONG` markers (constant delay)
- **Mobitz I**: Progressive `INCREASE` markers + dropped beat
- **Mobitz II**: Constant `LONG` + sudden drop
- **Third-Degree**: Independent P and R patterns

---

## 📊 Data Source

**Why Synthetic Data?**

Real AV block ECG data is rare and hard to obtain. We generated 100 synthetic sequences based on:

- Clinical guidelines: Hampton & Adlam (2019) - *The ECG Made Easy*
- Medical textbook: Surawicz & Knilans (2008) - *Chou's Electrocardiography*
- **Validated by:** Dr. Ahmed Bennani, Cardiologist at CHU Rabat (December 2024)

Future versions will use real ECG data from PhysioNet database.

---

## 💡 Key Findings

### What Worked Well ✅
- Perfect classification of Mobitz Type I (100% F1-score)
- Stack memory successfully captures progressive patterns
- Better than simple rule-based systems

### What Needs Improvement 📈
- Mobitz II and Third-Degree detection needs refinement
- Current features too simple to show full stack advantage
- Need richer symbolic encoding (PR interval values, temporal windows)

### Why PDA = FSA Accuracy?
Our simple features don't fully utilize the stack's power. This is honest research - we show what works and what doesn't!

---

## 🛠️ Technical Details

**PDA Definition:**  
M = (Q, Σ, Γ, δ, q₀, Z₀, F)

- **States (Q):** {q₀, q_first, q_mobitz1, q_mobitz2, q_third}
- **Input Alphabet (Σ):** {P, R, PRnormal, PRlong, PRincrease, drop}
- **Stack Alphabet (Γ):** {Z₀, LONG, PREV_PR, P_MARK, R_MARK}

**Complexity:**
- Time: O(n) - single pass through input
- Space: O(n) - stack grows with input

---

## 📚 References

1. Hampton & Adlam (2019). *The ECG Made Easy*. Elsevier.
2. Sipser (2012). *Introduction to the Theory of Computation*. Cengage Learning.
3. Goldberger et al. (2000). "PhysioNet: Components of a new research resource for complex physiologic signals." *Circulation*.
4. Hopcroft, Motwani & Ullman (2006). *Introduction to Automata Theory, Languages, and Computation*. Pearson.

---

## 🤝 Acknowledgments

- **Course Instructor** - For detailed feedback that improved this work
- **Dr. Ahmed Bennani** - Cardiologist, CHU Rabat (pattern validation)
- **PhysioNet** - Reference ECG database
- **Claude AI** - Code debugging and documentation assistance

---

## 📧 Contact

**Salma Bellaou** - salma.bellaou@um6p.ma  
**Asma Khamri** - asma.khamri@um6p.ma

Mohammed VI Polytechnic University  
College of Computing, Rabat Campus

---

## 📄 License

MIT License - Feel free to use this for educational purposes!

---

## 🚀 Future Work

- [ ] Test on real PhysioNet ECG data
- [ ] Add more discriminative features for Mobitz II and Third-Degree
- [ ] Implement probabilistic PDA variant
- [ ] Create web-based demo interface
- [ ] Extend to other arrhythmia types
