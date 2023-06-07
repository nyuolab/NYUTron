Step 1: Create a conda environment

```conda create -n nyutron python=3.8.13 cython```
```conda activate nyutron```

Step 2: Install packages with pip

```pip install -r documentation/requirements.txt```
```pip install -e .``` 

Step 3: Test installed package

```python tests/test_data_processing.py```

Congrats! You finish the install.

You can try some examples:
```cd examples```
```python 1_make_pretrain_dataset.py```
```python 2_pretrain.py```
```python 3_make_finetune_dataset.py```
```python 4_finetune.py```
```python 5_eval_ckpt.py```
```source 6_i2b2.sh```