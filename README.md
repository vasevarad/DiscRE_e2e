# DiscRE_e2e

First download the pretrained discre model, glove embeddings, and a dummy data file to test
```
python download_pretrained_models.py

```

Then run the example main file that takes in a message csv file, and returns the embeddings that are saved as a pickle.

```
python main.py
```

To change the format of the input file, change **prepare_input_file** function in **utils/TweeboParseUtils.py**

Note that the input to subsequent steps are always a file with one message per line. The other steps might throw an error if this is not the case.
