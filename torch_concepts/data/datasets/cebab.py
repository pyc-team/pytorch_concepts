from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch

class CEBaBDataset:
    def __init__(self, 
                 pre_trained_transformer='bert-base-uncased',
                 batch_size=32
    ):
        
        ds = load_dataset("CEBaB/CEBaB")
        self.batch_size = batch_size

        # get the relevant splits from the dataset (train observational contains only the original reviews which have not been modified by the annotators)
        ds_train = ds['train_observational']
        ds_val = ds['validation']
        ds_test = ds['test']

        # define concept names
        self.concept_names = ['food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']

        # convert to pandas
        ds_train = ds_train.to_pandas()
        ds_val = ds_val.to_pandas()
        ds_test = ds_test.to_pandas()

        # select only the relevant columns: review_majority, food_aspect_majority, ambiance_aspect_majority, service_aspect_majority, noise_aspect_majority
        ds_train = ds_train[['description', 'review_majority', 'food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']]
        ds_val = ds_val[['description', 'review_majority', 'food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']]
        ds_test = ds_test[['description', 'review_majority', 'food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']]

        # eliminate rows with missing values
        ds_train = ds_train.dropna()
        ds_val = ds_val.dropna()
        ds_test = ds_test.dropna()

        # apply the following mapping to the columns: food_aspect_majority, ambiance_aspect_majority, service_aspect_majority, noise_aspect_majority.
        # precisely, map the values 'Positive' to 2 and 'unknown' to 1 and 'Negative' to 0
        mapping = {'Positive': 2, 'unknown': 1, 'Negative': 0}
        
        # Encode any other value as 'no majority'
        for concept in self.concept_names:
            ds_train[concept] = ds_train[concept].apply(lambda x: mapping.get(x, 'no majority'))
            ds_val[concept] = ds_val[concept].apply(lambda x: mapping.get(x, 'no majority'))
            ds_test[concept] = ds_test[concept].apply(lambda x: mapping.get(x, 'no majority'))
        
        # eliminate all the rows for which the review_majority='no majority' 
        ds_train = ds_train[~ds_train.isin(['no majority']).any(axis=1)]
        ds_val = ds_val[~ds_val.isin(['no majority']).any(axis=1)]
        ds_test = ds_test[~ds_test.isin(['no majority']).any(axis=1)]

        # convert the review vote to int
        ds_train['review_majority'] = ds_train.apply(lambda x: int(x['review_majority']), axis=1)
        ds_val['review_majority'] = ds_val.apply(lambda x: int(x['review_majority']), axis=1)
        ds_test['review_majority'] = ds_test.apply(lambda x: int(x['review_majority']), axis=1)

        # instantiate the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_transformer)

        data_train = Dataset.from_pandas(ds_train)
        data_val = Dataset.from_pandas(ds_val)
        data_test = Dataset.from_pandas(ds_test)

        tokenized_train = data_train.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_train.column_names,
        )

        tokenized_val = data_val.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_val.column_names,
        )

        tokenized_test = data_test.map(
            self.preprocess_function,
            batched=True,
            remove_columns=data_test.column_names,
        )       

        self.tokenized_train = tokenized_train
        self.tokenized_val = tokenized_val
        self.tokenized_test = tokenized_test

    def preprocess_function(self, examples):
        model_inputs = self.tokenizer(
            examples["description"],
            truncation=True,
            padding = 'max_length'
        )
        model_inputs["review_majority"] = examples["review_majority"]
        
        # now add the concepts
        for concept in self.concept_names:
            model_inputs[concept] = examples[concept]

        return model_inputs

    def collator(self):
        data_collator = CustomDataCollator()
        loaded_train = DataLoader(
            self.tokenized_train, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=True
            ) 
        
        loaded_val = DataLoader(
            self.tokenized_val, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        loaded_test = DataLoader(
            self.tokenized_test, 
            collate_fn=data_collator, 
            batch_size=self.batch_size, 
            shuffle=False
            )
        
        return loaded_train, loaded_val, loaded_test

class CustomDataCollator:
    def __init__(self):
        pass
        
    def __call__(self, batch):

        # transform the batch into a tensor
        input_ids = torch.Tensor([example['input_ids'] for example in batch])
        token_type_ids = torch.Tensor([example['token_type_ids'] for example in batch])
        attention_mask = torch.Tensor([example['attention_mask'] for example in batch])
        labels = torch.Tensor([example['review_majority'] for example in batch])
        food = torch.Tensor([example['food_aspect_majority'] for example in batch])
        ambiance = torch.Tensor([example['ambiance_aspect_majority'] for example in batch])
        service = torch.Tensor([example['service_aspect_majority'] for example in batch])
        noise = torch.Tensor([example['noise_aspect_majority'] for example in batch])

        # concatenate the concepts in the same tensor
        concepts = torch.stack([food, ambiance, service, noise], dim=1)

        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'concepts': concepts
        }
    
def main():
    loader = CEBaBDataset('bert-base-uncased', 128)
    train_loader, _, _ = loader.collator()

    for batch in train_loader:
        print(batch['input_ids'])
        break

if __name__=="__main__":
    main()
