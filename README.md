# CS6910 Assignment 3
## Author : P.Sai Sree Ram CS22M076
### Instructions to create a model , train , predict CNN Network :
1. Install the required modules/libraries before running
2. Maintain proper folder structure of the dataset
```
base_dir = aksharantar_sampled/
base_dir = base_dir+lang+"/" 

train_file = base_dir+lang+"_train.csv"
val_file = base_dir+lang+"_valid.csv"
test_file = base_dir+lang+"_test.csv"

```
3. To To create a model 
```
if isattention == True :
    model = Attn2seq(input_size = len(latin_chars), output_size = len(lang_chars),
                embedding_size = embedding_size, hidden_size = hidden_layer_size,encoder_layer_size = encoder_layers,
                decoder_layer_size = encoder_layers,cell_type = cell_type,
                beam_width = 1,dropout= dropout,
                bidirectional =bidirectional ,learning_rate = learning_rate)

else :
    model = seq2seq(input_size = len(latin_chars), output_size = len(lang_chars),
                embedding_size = embedding_size, hidden_size = hidden_layer_size,encoder_layer_size = encoder_layers,
                decoder_layer_size = decoder_layers,cell_type = cell_type,
                beam_width = 1,dropout= dropout,
                bidirectional =bidirectional ,learning_rate = learning_rate)

```
4. To train the modal
```
trainer = pl.Trainer(max_epochs = 20)
trainer.fit(model, train_dataloader,val_dataloader)

```

5. To run the train.py
 ```
python train.py --wandb_entity myname --wandb_project myprojectname

python train.py -dp aksharantar_sampled/ -la hin -e 5 -b 128 -lr 0.0001 -cl GRU -bd True -at True -do 0.2 -el 3 -dl 3 -es 16 -hls 128
 ```
All the argumets are listed below 

| Name | Default Value | Description |
| :---: | :----------: | :--------|
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-dp`, `--dataset_path` | aksharantar_sampled | Give folder name with / |
| `-la`, `--lang` | hin | language name given in folder  |
| `-e`, `--epochs` | 1 | Number of epochs to train the model. |
| `-b`, `--batch_size` | 16 | Batch size used to train the model. |
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters |
| `-cl`, `--cell_type`| LSTM | choices: [LSTM, RNN, GRU] |
| `-bd`, `--bidirectional` | True |  whether the RNN layers are bidirectional or not choices = [True , False] |
| `-at`, `--isattention` | True | To add attention layer |
| `-do`, `--dropout` | 0.1 | Value of dropout |
| `-el`, `--encoder_layers` | 3 | The number of layers in the encoder |
| `-dl`, `--decoder_layers` | 2 | The number of layers in the decoder |
| `-es`, `--embedding_size` | 64 | The size of the embedding vectors for each token |
| `-hls`, `--hidden_layer_size` | 128 | The size of the hidden state of the RNN cells in the encoder and decoder |

