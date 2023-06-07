Here we document the scripts we used for all experiments in our paper:

**Main Figures**

* Fig 2b,c - Exp 1,2,3

* Fig 3a -  Exp 5

* Fig 3b - Exp 2, 3

* Fig 3c,d -  Exp 1,2

* Fig 4a - Exp 6, Exp 5

* Fig 4b -  annotations from residents

Figure plotting script and data for plotting can be found at  [visualization/main_figures](../visualization/main_figures)

---

**Extended Data Figures**

* ED Fig 1a - Exp 5

* ED Fig 1b - Exp 2

* ED Fig 2a - Exp 3

* ED Fig 2b - Exp 7

* ED Fig 4 - Exp 2, different configs for MIMIC data (refer [sensitivity score](https://github.com/nyuolab/Model_Sensitivity) for data processing)

* ED Fig 5a,b, ED Fig 6a,b, Fig 8b - Exp 5, using different stratified data

* ED Fig 8c - Exp 5 (for saving predicted probs and actual label)

---

**SI**

* SI Fig 1 - in ```visualization```

* SI 2.2 different types of notes - Exp 2, different data configs

* Table 2. Exp 3, different data

---

**Experiments**

Exp 1. Pretrain NYUTron, NYUTron-Manhattan, NYUTron-Brooklyn 
---

* scripts: [1_make_pretrain_dataset.py](../examples/1_make_pretrain_dataset.py), [2_pretrain.py](../examples/2_pretrain.py)
* [data configs](../examples/configs/pretrain_data_configs)
* [train configs](../examples/configs/pretrain_configs)
* [step-by-step guide](pretrain.md)

---

Exp 2. Finetune NYUTron / other LMs for readmission prediction, mortality prediction, insurnace denial prediction, LOS prediction and comorbidity preidtion
---

* scripts: [3_make_finetune_dataset.py](../examples/3_make_finetune_dataset.py), [4_finetune.py](../examples/4_finetune.py)
* [data configs](../examples/configs/finetune_data_configs)
* [train configs](../examples/configs/finetune_configs)
* [step-by-step guide](finetune.md)

Note:
- insurance denial dataset (which has 2 types of notes) requires different preprocessing (see [examples/supplementary/insurance_denial](../examples/supplementary/insurance_denial)).
- unlike the other downstream tasks, comorbidity prediction has 4 labels (rather than 2 labels). The current python script is compatible with both modes. To see a toy example for multi-class classification, use the commented option for ```data_opt``` in [3_make_finetune_dataset.py](../examples/3_make_finetune_dataset.py) and use the commented hydra config (above ```def finetune```) in [4_finetune.py](../examples/4_finetune.py).

---

Exp 3. Baselines: Structured Data + XGB and TF-IDF + XGB 
---

```cd baselines```

```source 2_run_struct_baseline.sh``` (configs at [baselines/configs](../baselines/configs))

```source 3_run_text_baseline.sh```

Note:
* ```struct_baseline``` requires GPU for categorical features. 
* Configs for structured baseline can be editted in the yaml file. Configs for text baseline can be editted in the scripts.

---

Exp 4. Generate redcap questions
---

We provided some key functions in the script [baselines/4_redcap_processing.py](baselines/4_redcap_processing.py).

```generate_questions``` takes in two pandas dataframe (with positivly and negative labelled discharge notes) and generate redcap questions for doctors. 

```update_redcap_survey``` takes in a csv downloaded from redcap (we got this csv from our research coordinator, who kindly helped us manually configure all the questions and response fields using redcap), and added a section header and an additonal multiple choice option.

---

Exp 5. Evaluate auc (including stratified eval)
---

```cd examples```

```python 5_eval_ckpt.py```

For plotting calibration, see [8cd_calibration_curve.py](visualization/extended_data_and_SI/8cd_calibration_curve.py)

For stratified eval, need to load data for different strats (e.g., department, age, race) and potentially additional metric (e.g., roc_auc).

---

Exp 6. NYUTriton 
---

We used a custom triton inference engine for prospective test.
For details, see accompnaying repo [NYUTriton](https://github.com/nyuolab/NYUTriton_open).

---

Exp 7. i2b2 finetuning
---

```cd examples/utils```

```python tokenize_i2b2.py```

```cd ..```

```source 6_i2b2.sh```

More details about preprocessing can be found at [i2b2_2012_preprocessing](https://github.com/nyuolab/i2b2_2012_preprocessing).
