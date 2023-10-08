# GFlowNet fine-tuning for story infilling, subjectivity classification and integer arithmetic

To install the dependencies required to run the code in this directory use `pip install -r requirements.txt`



### Story infilling

For the infilling task please download the ROC Stories dataset by filling the form at https://cs.rochester.edu/nlp/rocstories/. Once you receive the email, download  `ROCStories_winter2017 - ROCStories_winter2017.csv`, rename it to `stories.csv` and place it in the `data` directory. We also need to fine-tune a GPT-2 Large model on this dataset. First convert the dataset to the required format using `make_sft_data.py` and then run the `sft.py` with the appropriate `--output_dir` parameter. 

Command to run GFlowNet fine-tuning for story infilling:
```bash
python main.py +experiment=infilling model.name="<path of fine-tuned gpt2>"
```

To compute the metrics you can load the checkpoint stored during training with the `eval_infill.py` script.

```bash
python eval_infill.py --batch_size 1024 --temp 0.9  --max_eval_len 25 --load_checkpoint_path "<path_here>"
```

### Subjectivity Classification

To run the GFlowNet fine-tuning experiment, run `Subj_obj_GFN_FT_E_step.ipynb` (for the M step see `Subj_obj_GFN_FT_M_step.ipynb`)

### Arithmetic with tool use

We include the datasets used for the arithmetic task and to generate the dataset we used `data_integer_arithmetic.ipynb`.


Command to run GFlowNet fine-tuning for arithmetic with calculator:
```bash
python main.py +experiment=arithmetic_varying 
```

To compute the metrics you can load the checkpoint stored during training with the `eval_arithmetic.py` script.

```bash
python prompting.py +experiment=arithmetic_varying test_only=True max_eval_len=22 dataset.test_path=data/arithmetic_with_tool/1digit_3_op+-_test.json use_tools=True limit_capability=2 reward.reward_config.prompt_data.num_points=0 data.num_test=10000 load_checkpoint_path="<path_here>"
```

Please refer to `configs/rationale_buffer.yaml` for the hyperparameters.