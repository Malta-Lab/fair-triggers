# fair-triggers
* This repository is divided into two parts:
  * The first is 

***
## Objective:
* Use triggers with language modeling and other tasks to reduce bias.

## Datasets:
* Bias in Bios: https://github.com/microsoft/biosbias
***
## Related Work:
* Our work is based on other two papers:
  * [Universal Triggers](https://arxiv.org/abs/1908.07125)
    * Which the code is heavily inspired.
  * [Bias Triggers](https://arxiv.org/abs/2005.00268)


## Running the Method:

### Installing dependencies
```
pip install -r requirements.txt
```
### Downloading Dataset
* Go to the [dataset repository](https://github.com/microsoft/biosbias)
* Install it
* Install warcio, pebble (python packages)
* Run ./recreate.sh
  * There may be a problem with the download link, to fix it just change the COMMON_CRAWL_URL in the downloading script to 'https://data.commoncrawl.org/'.

### Running BERT training
* Our experiments contains two steps, the first is focused on creating a biased model. Se that we can further create a trigger to debiasing the model.
#### The code to train the model is in the folder:
  * ``` ./train_bert_classifier```

#### To train the BERT classifier:
  * ```python bert_training.py -exp 5_class_model --labels rapper nurse surgeon dj psychologist```
    * ```-exp``` sets just the name of the experiment
    * ```--labels``` defines the classes. Not using this parameter will train a model without class restriction.
  * This training will generate a ```checkpoints``` folder, with a subfolder with the name of the experiment.

#### To evaluate the BERT classifier:
* ```python evaluate.py -exp 5_class_model --labels rapper nurse surgeon dj psychologist```
  * Contains the same configurations, but this only load the model and evaluate it.

#### Storing BERT results:
* ```store_results.py --ckpt ../checkpoints/5_class_model/best_model --labels rapper nurse surgeon dj psychologist```
  * We set the checkpoint for the model weights, and keep the number of labels we are using.
  * Besides, we can set a trigger by the argument ```--trigger```, using the numbers of the tokens:
    * ```store_results.py --ckpt ../checkpoints/5_class_model/best_model --labels rapper nurse surgeon dj psychologist --trigger 24224  8838 27464 14502 21777 10627```
  * We use this command to test how a model react to the triggers and compare with the base model.

### Generating BERT Adversarial Trigger
* To create the trigger for a BERT model.
  * ```python create_bert_token.py --model ./checkpoints/5_class_model/best_model --label nurse```
    * Where ```--model``` receive the checkpoint
    * And ```--label``` receive which label we want to create the trigger for
    * A last possible paramenter is the ```--all_labels rapper nurse surgeon dj psychologist```, setting which classes the model work with, useful when the problem contains less classes than we have on the dataset.

### Generating GPT-2 Adversarial Trigger
* We can also generate an adversarial trigger for GPT model, by running the python script:
  * ```python create_adv_token.py -rf targets/racist.txt```
    * Where ```-rf``` is the repository where the intent files are located. 
    * To create a trigger to induce a specific behavior we just need to use a txt file with sample sentences that contain such behavior.
*  Also, we can generate a word cloud based on the generated sample of text:
   *  ```python word_cloud.py --input_file <file_containing_samples> --output_file <file_to_save_figure> --trigger "<trigger>"```