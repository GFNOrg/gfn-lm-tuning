import numpy as np
import pandas as pd
import json
from hydra.utils import to_absolute_path

def load_dataset(config, tokenizer, append_prompt_test=None):
    if config.dataset.name == "integer":
        train_path = to_absolute_path(config.dataset.train_path)
        hw_path = to_absolute_path(config.dataset.hw_path)
        test_path = to_absolute_path(config.dataset.test_path)
        
        data_train = json.load(open(train_path, 'r'))
        data_hw = json.load(open(hw_path, 'r'))
        data_test = json.load(open(test_path, 'r'))

        train_queries = []
        train_rationales = []
        train_sols = []
        train_weight = []

        test_queries = []
        test_num_sols = []

        for sample in data_train[:config.dataset.num_train]:
            train_queries.append(sample['str_query'])
            train_rationales.append(sample['str_rationale'])
            train_sols.append(sample['str_sol'])
            train_weight.append(1)
        print("train_queries", len(train_queries))
        for sample in data_hw[:config.dataset.num_hw]:
            train_queries.append(sample['str_query'])
            train_rationales.append(None)
            train_sols.append(sample['str_sol'])
            train_weight.append(0.2)

        train_weight = np.array(train_weight)
        train_weight = train_weight / train_weight.sum()

        for sample in data_test[:config.dataset.num_test]:
            if append_prompt_test:
                test_queries.append(append_prompt_test + sample['str_query'])
            else:
                test_queries.append(sample['str_query'])
            if "num_sol" in sample:
                test_num_sols.append(sample['num_sol'])
            else:
                test_num_sols = None
        
        encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
        encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
        encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
   
    elif config.dataset.name == "stories":
        df = pd.read_csv(to_absolute_path(config.dataset.path))
        data = []

        for index in range(100, 1000):
            row = df.iloc[index]
            story = {}
            story["str_query"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
            story["str_sol"] = config.dataset.sol_prefix + row["sentence5"]
            story["str_rationale"] = " " + row["sentence4"][:-1]
            # print(row['sentence1'])
            # if index == config.dataset.max_data:
            #     break
            data.append(story)
        
        train_queries = []
        train_rationales = []
        train_sols = []
        train_weight = []

        test_queries = []
        test_num_sols = []
        test_sols = []

        for i, sample in enumerate(data):
            train_queries.append(sample['str_query'])
            if config.dataset.use_rationales and i < config.dataset.max_data:
                train_rationales.append(sample['str_rationale'])
            else:
                train_rationales.append(None)
            train_sols.append(sample['str_sol'])
            train_weight.append(1)

        train_weight = np.array(train_weight)
        train_weight = train_weight / train_weight.sum()

        for sample in data:
            test_queries.append(sample['str_query'])
            if "num_sol" in sample:
                test_num_sols.append(sample['num_sol'])
            else:
                test_num_sols = None
            test_sols.append(sample['str_sol'])
        
        encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
        encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
        encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
    
    elif config.dataset.name == "stories_in_prompt_2":
        df = pd.read_csv(to_absolute_path(config.dataset.path))
        data = []

        for index in range(100, 1000):
            row = df.iloc[index]
            story = {}
            story["str_query"] = "Beginning: " + row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
            story["str_query"] += "\nEnding: " + row["sentence5"] + "\nMiddle:"
            story["str_sol"] =  row["sentence5"]
            story["str_rationale"] = " " + row["sentence4"]
            story["beginning"] = row["sentence1"] + " " + row["sentence2"] + " " + row["sentence3"]
            # print(row['sentence1'])
            # if index == config.dataset.max_data:
            #     break
            data.append(story)
        
        train_queries = []
        train_rationales = []
        train_sols = []
        train_weight = []
        train_beginning = []

        test_queries = []
        test_num_sols = []
        test_sols = []
        test_beginning = []

        for i, sample in enumerate(data):
            train_queries.append(sample['str_query'])
            if config.dataset.use_rationales and i < config.dataset.max_data:
                train_rationales.append(sample['str_rationale'])
            else:
                train_rationales.append(None)
            train_sols.append(sample['str_sol'])
            train_weight.append(1)
            train_beginning.append(sample['beginning'])

        train_weight = np.array(train_weight)
        train_weight = train_weight / train_weight.sum()

        for sample in data:
            test_queries.append(sample['str_query'])
            if "num_sol" in sample:
                test_num_sols.append(sample['num_sol'])
            else:
                test_num_sols = None
            test_sols.append(sample['str_sol'])
            test_beginning.append(sample['beginning'])
        
        encoded_train_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_queries]
        encoded_train_sols = [tokenizer(answer, return_tensors='pt')['input_ids'].cuda() for answer in train_sols]
        encoded_test_queries = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_queries]
        encoded_train_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in train_beginning]
        encoded_test_beginning = [tokenizer(query, return_tensors='pt')['input_ids'].cuda() for query in test_beginning]
        return {
            "train_queries": train_queries,
            "train_rationales": train_rationales,
            "train_sols": train_sols,
            "test_sols": test_sols,
            "train_beginning": train_beginning,
            "test_beginning": test_beginning,
            "encoded_train_queries": encoded_train_queries,
            "encoded_train_sols": encoded_train_sols,
            "test_queries": test_queries,
            "encoded_test_queries": encoded_test_queries,
            "encoded_train_beginning": encoded_train_beginning,
            "encoded_test_beginning": encoded_test_beginning,
            "train_weight": train_weight if config.dataset.name == "integer" else None,
            "test_num_sols": test_num_sols if config.dataset.name == "integer" else None,
        }
    else:
        raise NotImplementedError("Please select a valid dataset.")

    return {
        "train_queries": train_queries,
        "train_rationales": train_rationales,
        "train_sols": train_sols,
        "test_sols": test_sols if config.dataset.name == "stories" else None,
        "encoded_train_queries": encoded_train_queries,
        "encoded_train_sols": encoded_train_sols,
        "test_queries": test_queries,
        "encoded_test_queries": encoded_test_queries,
        "train_weight": train_weight if config.dataset.name == "integer" else None,
        "test_num_sols": test_num_sols if config.dataset.name == "integer" else None,
    }