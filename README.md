# (Yet Another) Tensorflow ChatRNN
Yeah I know, there are like million versions of Tensorflow ChatRNN. This is my version.

### Training:

    python train.py --data TEXT_PATH --save CHECKPOINTS_FOLDER

### Evalling:

    python eval.py --path CHECKPOINTS_FOLDER --prime PRIME

**or**

    python eval.py --path CHECKPOINTS_FOLDER --loop 1

## For example - Shakespeare
Yay, another machine learning which comes with tiny Shakespeare dataset.


**Train it (It's already half-trained so you don't really have to do this step):**

    python train.py --data shakespeare/input.txt --save shakespeare/checkpoints/

**Use it:**

    python eval.py --path shakespeare/checkpoints/ --loop 1

