
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.248091603053435
    Number of lines: 4257
    Average number of words in each line: 11.50434578341555
    
    The sentences 0 to 10:
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    
    

## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab_to_int = {}
    int_to_vocab = {}
    
    sorted_text = sorted(set(text))
    vocab_to_int = {vocab:counter for counter, vocab in enumerate(sorted_text)}
    int_to_vocab = dict(enumerate(sorted_text))
    
    return (vocab_to_int, int_to_vocab)
#     vocab = set(text)
#     # Use comprenhension lists to build our dictionaries.
#     vocab_to_int = {word:idx for idx, word in enumerate(vocab)}
#     int_to_vocab = {idx:word for idx, word in enumerate(vocab)}
#     return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    token_to_string = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_Mark||',
        '?' : '||Question_Mark||',
        '(' : '||Left_Parentheses||',
        ')' : '||Right_Parentheses||',
        '--' : '||Dash||',
        '\n' : '||Return||'
    }
    return token_to_string

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    

    C:\Users\ikc\Anaconda3\envs\dlnd\lib\site-packages\ipykernel_launcher.py:14: UserWarning: No GPU found. Please use a GPU to train your neural network.
      
    

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    return (inputs, targets, learning_rate)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed
    

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Add dropout to the cell
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = 0.5)
    
    # Stack up multiple LSTM laters, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([drop] * 3)
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name = 'initial_state')
    
    return (cell, initial_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed
    

### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name = 'final_state')
    
    return (outputs, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed
    

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    inputs = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, inputs)
#     weights = tf.truncated_normal_initializer(stddev = 0.1)
#     biases = tf.zeros_initializer()
#     # activation_fn: Activation function. The default value is a ReLU function. Explicitly set it to None to skip it and maintain a linear activation.
#     logits = tf.contrib.layers.fully_connected(outputs,
#                                                vocab_size,
#                                                weights_initializer = weights,
#                                                biases_initializer = biases,
#                                                activation_fn = None)
    logits = tf.layers.dense(outputs, vocab_size, activation= None)

    
    return (logits, final_state)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed
    

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For example, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    # Calculate the number of batches
    n_batches = np.floor(len(int_text)//(batch_size * seq_length))
    print('Length of int_text is', len(int_text))
    print('Number of batches is ', n_batches)
    print('Batch size is ', batch_size)
    print('Sequence length is ', seq_length)
    
    # Length of the int_text after dropping the last batch
    len_total_word = int(n_batches * batch_size * seq_length)
    
    
    # Truncated the given array
    int_text = int_text[:len_total_word]
#     print(int_text)
    
    x = np.array(int_text)
    y = np.zeros_like(x)
    y[:-1], y[-1] = x[1:], x[0]
    
    input_batches = x
    target_batches = y
    
#     # Initial input and target batches with the enough characters to make full batches    
#     input_batches = np.array(int_text)
#     roll_int_text = np.roll(int_text, -1)
#     target_batches = np.array(roll_int_text[:len_total_word]) # Shift 1 element to right for target
    
    # Reshape into batch_size rows
    input_batches = input_batches.reshape(batch_size, -1)
    target_batches = target_batches.reshape(batch_size, -1)
    
    # Split the word with calculated number of batches
    input_batches = np.split(input_batches, n_batches, 1)
    target_batches = np.split(target_batches, n_batches, 1)
    
    # Zip and pack into the list
    list_batches = list(zip(input_batches, target_batches))
#     print(list_batches[-1])
    
    
    # Convert to Numpy array
    ret_batches = np.array(list_batches)
    print('The shape of ret_batches is ', ret_batches.shape)
    
    print('The shape of first element in the batch is ', ret_batches[0].shape)
    print('The shape of second element in the batch is ', ret_batches[1].shape)
    
    
#    n_batches = len(int_text) // (batch_size * seq_length)
#     result = []
#     for i in range(n_batches):
#         inputs = []
#         targets = []
#         for j in range(batch_size):
#             idx = i * seq_length + j * seq_length
#             inputs.append(int_text[idx:idx + seq_length])
#             targets.append(int_text[idx + 1:idx + seq_length + 1])
#         result.append([inputs, targets])
#     return np.array(result)
    
    
    return ret_batches


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Length of int_text is 5000
    Number of batches is  7.0
    Batch size is  128
    Sequence length is  5
    The shape of ret_batches is  (7, 2, 128, 5)
    The shape of first element in the batch is  (2, 128, 5)
    The shape of second element in the batch is  (2, 128, 5)
    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 100
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 512
# Embedding Dimension Size
embed_dim = 256
# Sequence Length
seq_length = 16
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 10

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forums](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
import time
```


```python
t0_train = time.time()
```


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Length of int_text is 69100
    Number of batches is  33.0
    Batch size is  128
    Sequence length is  16
    The shape of ret_batches is  (33, 2, 128, 16)
    The shape of first element in the batch is  (2, 128, 16)
    The shape of second element in the batch is  (2, 128, 16)
    Epoch   0 Batch    0/33   train_loss = 8.822
    Epoch   0 Batch   10/33   train_loss = 6.560
    Epoch   0 Batch   20/33   train_loss = 6.495
    Epoch   0 Batch   30/33   train_loss = 6.472
    Epoch   1 Batch    7/33   train_loss = 6.214
    Epoch   1 Batch   17/33   train_loss = 6.183
    Epoch   1 Batch   27/33   train_loss = 6.237
    Epoch   2 Batch    4/33   train_loss = 6.073
    Epoch   2 Batch   14/33   train_loss = 6.235
    Epoch   2 Batch   24/33   train_loss = 6.122
    Epoch   3 Batch    1/33   train_loss = 6.002
    Epoch   3 Batch   11/33   train_loss = 5.977
    Epoch   3 Batch   21/33   train_loss = 6.183
    Epoch   3 Batch   31/33   train_loss = 6.078
    Epoch   4 Batch    8/33   train_loss = 6.080
    Epoch   4 Batch   18/33   train_loss = 6.087
    Epoch   4 Batch   28/33   train_loss = 6.061
    Epoch   5 Batch    5/33   train_loss = 6.123
    Epoch   5 Batch   15/33   train_loss = 6.001
    Epoch   5 Batch   25/33   train_loss = 6.035
    Epoch   6 Batch    2/33   train_loss = 5.849
    Epoch   6 Batch   12/33   train_loss = 5.961
    Epoch   6 Batch   22/33   train_loss = 6.011
    Epoch   6 Batch   32/33   train_loss = 6.041
    Epoch   7 Batch    9/33   train_loss = 5.836
    Epoch   7 Batch   19/33   train_loss = 5.889
    Epoch   7 Batch   29/33   train_loss = 5.875
    Epoch   8 Batch    6/33   train_loss = 5.828
    Epoch   8 Batch   16/33   train_loss = 5.814
    Epoch   8 Batch   26/33   train_loss = 5.790
    Epoch   9 Batch    3/33   train_loss = 5.736
    Epoch   9 Batch   13/33   train_loss = 5.816
    Epoch   9 Batch   23/33   train_loss = 5.816
    Epoch  10 Batch    0/33   train_loss = 5.696
    Epoch  10 Batch   10/33   train_loss = 5.864
    Epoch  10 Batch   20/33   train_loss = 5.737
    Epoch  10 Batch   30/33   train_loss = 5.695
    Epoch  11 Batch    7/33   train_loss = 5.762
    Epoch  11 Batch   17/33   train_loss = 5.723
    Epoch  11 Batch   27/33   train_loss = 5.731
    Epoch  12 Batch    4/33   train_loss = 5.596
    Epoch  12 Batch   14/33   train_loss = 5.753
    Epoch  12 Batch   24/33   train_loss = 5.627
    Epoch  13 Batch    1/33   train_loss = 5.456
    Epoch  13 Batch   11/33   train_loss = 5.441
    Epoch  13 Batch   21/33   train_loss = 5.691
    Epoch  13 Batch   31/33   train_loss = 5.556
    Epoch  14 Batch    8/33   train_loss = 5.573
    Epoch  14 Batch   18/33   train_loss = 5.526
    Epoch  14 Batch   28/33   train_loss = 5.525
    Epoch  15 Batch    5/33   train_loss = 5.579
    Epoch  15 Batch   15/33   train_loss = 5.397
    Epoch  15 Batch   25/33   train_loss = 5.397
    Epoch  16 Batch    2/33   train_loss = 5.248
    Epoch  16 Batch   12/33   train_loss = 5.377
    Epoch  16 Batch   22/33   train_loss = 5.392
    Epoch  16 Batch   32/33   train_loss = 5.399
    Epoch  17 Batch    9/33   train_loss = 5.185
    Epoch  17 Batch   19/33   train_loss = 5.167
    Epoch  17 Batch   29/33   train_loss = 5.174
    Epoch  18 Batch    6/33   train_loss = 5.115
    Epoch  18 Batch   16/33   train_loss = 5.071
    Epoch  18 Batch   26/33   train_loss = 5.073
    Epoch  19 Batch    3/33   train_loss = 4.973
    Epoch  19 Batch   13/33   train_loss = 5.073
    Epoch  19 Batch   23/33   train_loss = 5.098
    Epoch  20 Batch    0/33   train_loss = 4.930
    Epoch  20 Batch   10/33   train_loss = 5.057
    Epoch  20 Batch   20/33   train_loss = 4.940
    Epoch  20 Batch   30/33   train_loss = 4.958
    Epoch  21 Batch    7/33   train_loss = 4.940
    Epoch  21 Batch   17/33   train_loss = 4.925
    Epoch  21 Batch   27/33   train_loss = 4.955
    Epoch  22 Batch    4/33   train_loss = 4.785
    Epoch  22 Batch   14/33   train_loss = 4.879
    Epoch  22 Batch   24/33   train_loss = 4.823
    Epoch  23 Batch    1/33   train_loss = 4.594
    Epoch  23 Batch   11/33   train_loss = 4.616
    Epoch  23 Batch   21/33   train_loss = 4.831
    Epoch  23 Batch   31/33   train_loss = 4.734
    Epoch  24 Batch    8/33   train_loss = 4.712
    Epoch  24 Batch   18/33   train_loss = 4.738
    Epoch  24 Batch   28/33   train_loss = 4.745
    Epoch  25 Batch    5/33   train_loss = 4.740
    Epoch  25 Batch   15/33   train_loss = 4.641
    Epoch  25 Batch   25/33   train_loss = 4.655
    Epoch  26 Batch    2/33   train_loss = 4.506
    Epoch  26 Batch   12/33   train_loss = 4.650
    Epoch  26 Batch   22/33   train_loss = 4.670
    Epoch  26 Batch   32/33   train_loss = 4.667
    Epoch  27 Batch    9/33   train_loss = 4.492
    Epoch  27 Batch   19/33   train_loss = 4.496
    Epoch  27 Batch   29/33   train_loss = 4.532
    Epoch  28 Batch    6/33   train_loss = 4.454
    Epoch  28 Batch   16/33   train_loss = 4.454
    Epoch  28 Batch   26/33   train_loss = 4.466
    Epoch  29 Batch    3/33   train_loss = 4.325
    Epoch  29 Batch   13/33   train_loss = 4.489
    Epoch  29 Batch   23/33   train_loss = 4.519
    Epoch  30 Batch    0/33   train_loss = 4.367
    Epoch  30 Batch   10/33   train_loss = 4.484
    Epoch  30 Batch   20/33   train_loss = 4.391
    Epoch  30 Batch   30/33   train_loss = 4.413
    Epoch  31 Batch    7/33   train_loss = 4.408
    Epoch  31 Batch   17/33   train_loss = 4.397
    Epoch  31 Batch   27/33   train_loss = 4.452
    Epoch  32 Batch    4/33   train_loss = 4.247
    Epoch  32 Batch   14/33   train_loss = 4.400
    Epoch  32 Batch   24/33   train_loss = 4.338
    Epoch  33 Batch    1/33   train_loss = 4.093
    Epoch  33 Batch   11/33   train_loss = 4.149
    Epoch  33 Batch   21/33   train_loss = 4.302
    Epoch  33 Batch   31/33   train_loss = 4.245
    Epoch  34 Batch    8/33   train_loss = 4.192
    Epoch  34 Batch   18/33   train_loss = 4.223
    Epoch  34 Batch   28/33   train_loss = 4.270
    Epoch  35 Batch    5/33   train_loss = 4.256
    Epoch  35 Batch   15/33   train_loss = 4.128
    Epoch  35 Batch   25/33   train_loss = 4.164
    Epoch  36 Batch    2/33   train_loss = 4.072
    Epoch  36 Batch   12/33   train_loss = 4.163
    Epoch  36 Batch   22/33   train_loss = 4.210
    Epoch  36 Batch   32/33   train_loss = 4.196
    Epoch  37 Batch    9/33   train_loss = 4.074
    Epoch  37 Batch   19/33   train_loss = 4.050
    Epoch  37 Batch   29/33   train_loss = 4.083
    Epoch  38 Batch    6/33   train_loss = 4.042
    Epoch  38 Batch   16/33   train_loss = 4.045
    Epoch  38 Batch   26/33   train_loss = 4.009
    Epoch  39 Batch    3/33   train_loss = 3.945
    Epoch  39 Batch   13/33   train_loss = 4.077
    Epoch  39 Batch   23/33   train_loss = 4.023
    Epoch  40 Batch    0/33   train_loss = 3.939
    Epoch  40 Batch   10/33   train_loss = 4.050
    Epoch  40 Batch   20/33   train_loss = 3.966
    Epoch  40 Batch   30/33   train_loss = 4.021
    Epoch  41 Batch    7/33   train_loss = 3.984
    Epoch  41 Batch   17/33   train_loss = 4.005
    Epoch  41 Batch   27/33   train_loss = 4.044
    Epoch  42 Batch    4/33   train_loss = 3.851
    Epoch  42 Batch   14/33   train_loss = 3.958
    Epoch  42 Batch   24/33   train_loss = 3.966
    Epoch  43 Batch    1/33   train_loss = 3.741
    Epoch  43 Batch   11/33   train_loss = 3.833
    Epoch  43 Batch   21/33   train_loss = 3.912
    Epoch  43 Batch   31/33   train_loss = 3.853
    Epoch  44 Batch    8/33   train_loss = 3.834
    Epoch  44 Batch   18/33   train_loss = 3.857
    Epoch  44 Batch   28/33   train_loss = 3.868
    Epoch  45 Batch    5/33   train_loss = 3.859
    Epoch  45 Batch   15/33   train_loss = 3.788
    Epoch  45 Batch   25/33   train_loss = 3.801
    Epoch  46 Batch    2/33   train_loss = 3.694
    Epoch  46 Batch   12/33   train_loss = 3.809
    Epoch  46 Batch   22/33   train_loss = 3.787
    Epoch  46 Batch   32/33   train_loss = 3.814
    Epoch  47 Batch    9/33   train_loss = 3.664
    Epoch  47 Batch   19/33   train_loss = 3.672
    Epoch  47 Batch   29/33   train_loss = 3.687
    Epoch  48 Batch    6/33   train_loss = 3.635
    Epoch  48 Batch   16/33   train_loss = 3.652
    Epoch  48 Batch   26/33   train_loss = 3.634
    Epoch  49 Batch    3/33   train_loss = 3.559
    Epoch  49 Batch   13/33   train_loss = 3.696
    Epoch  49 Batch   23/33   train_loss = 3.672
    Epoch  50 Batch    0/33   train_loss = 3.598
    Epoch  50 Batch   10/33   train_loss = 3.653
    Epoch  50 Batch   20/33   train_loss = 3.613
    Epoch  50 Batch   30/33   train_loss = 3.664
    Epoch  51 Batch    7/33   train_loss = 3.601
    Epoch  51 Batch   17/33   train_loss = 3.611
    Epoch  51 Batch   27/33   train_loss = 3.674
    Epoch  52 Batch    4/33   train_loss = 3.502
    Epoch  52 Batch   14/33   train_loss = 3.599
    Epoch  52 Batch   24/33   train_loss = 3.580
    Epoch  53 Batch    1/33   train_loss = 3.379
    Epoch  53 Batch   11/33   train_loss = 3.469
    Epoch  53 Batch   21/33   train_loss = 3.555
    Epoch  53 Batch   31/33   train_loss = 3.538
    Epoch  54 Batch    8/33   train_loss = 3.470
    Epoch  54 Batch   18/33   train_loss = 3.527
    Epoch  54 Batch   28/33   train_loss = 3.583
    Epoch  55 Batch    5/33   train_loss = 3.511
    Epoch  55 Batch   15/33   train_loss = 3.466
    Epoch  55 Batch   25/33   train_loss = 3.497
    Epoch  56 Batch    2/33   train_loss = 3.418
    Epoch  56 Batch   12/33   train_loss = 3.513
    Epoch  56 Batch   22/33   train_loss = 3.457
    Epoch  56 Batch   32/33   train_loss = 3.504
    Epoch  57 Batch    9/33   train_loss = 3.398
    Epoch  57 Batch   19/33   train_loss = 3.431
    Epoch  57 Batch   29/33   train_loss = 3.481
    Epoch  58 Batch    6/33   train_loss = 3.487
    Epoch  58 Batch   16/33   train_loss = 3.469
    Epoch  58 Batch   26/33   train_loss = 3.470
    Epoch  59 Batch    3/33   train_loss = 3.359
    Epoch  59 Batch   13/33   train_loss = 3.433
    Epoch  59 Batch   23/33   train_loss = 3.377
    Epoch  60 Batch    0/33   train_loss = 3.335
    Epoch  60 Batch   10/33   train_loss = 3.392
    Epoch  60 Batch   20/33   train_loss = 3.335
    Epoch  60 Batch   30/33   train_loss = 3.361
    Epoch  61 Batch    7/33   train_loss = 3.331
    Epoch  61 Batch   17/33   train_loss = 3.363
    Epoch  61 Batch   27/33   train_loss = 3.349
    Epoch  62 Batch    4/33   train_loss = 3.213
    Epoch  62 Batch   14/33   train_loss = 3.250
    Epoch  62 Batch   24/33   train_loss = 3.291
    Epoch  63 Batch    1/33   train_loss = 3.112
    Epoch  63 Batch   11/33   train_loss = 3.207
    Epoch  63 Batch   21/33   train_loss = 3.259
    Epoch  63 Batch   31/33   train_loss = 3.200
    Epoch  64 Batch    8/33   train_loss = 3.186
    Epoch  64 Batch   18/33   train_loss = 3.172
    Epoch  64 Batch   28/33   train_loss = 3.260
    Epoch  65 Batch    5/33   train_loss = 3.239
    Epoch  65 Batch   15/33   train_loss = 3.202
    Epoch  65 Batch   25/33   train_loss = 3.176
    Epoch  66 Batch    2/33   train_loss = 3.074
    Epoch  66 Batch   12/33   train_loss = 3.196
    Epoch  66 Batch   22/33   train_loss = 3.182
    Epoch  66 Batch   32/33   train_loss = 3.230
    Epoch  67 Batch    9/33   train_loss = 3.119
    Epoch  67 Batch   19/33   train_loss = 3.064
    Epoch  67 Batch   29/33   train_loss = 3.104
    Epoch  68 Batch    6/33   train_loss = 3.100
    Epoch  68 Batch   16/33   train_loss = 3.143
    Epoch  68 Batch   26/33   train_loss = 3.168
    Epoch  69 Batch    3/33   train_loss = 3.143
    Epoch  69 Batch   13/33   train_loss = 3.196
    Epoch  69 Batch   23/33   train_loss = 3.101
    Epoch  70 Batch    0/33   train_loss = 3.110
    Epoch  70 Batch   10/33   train_loss = 3.151
    Epoch  70 Batch   20/33   train_loss = 3.164
    Epoch  70 Batch   30/33   train_loss = 3.229
    Epoch  71 Batch    7/33   train_loss = 3.174
    Epoch  71 Batch   17/33   train_loss = 3.162
    Epoch  71 Batch   27/33   train_loss = 3.162
    Epoch  72 Batch    4/33   train_loss = 3.005
    Epoch  72 Batch   14/33   train_loss = 3.075
    Epoch  72 Batch   24/33   train_loss = 3.098
    Epoch  73 Batch    1/33   train_loss = 2.962
    Epoch  73 Batch   11/33   train_loss = 2.982
    Epoch  73 Batch   21/33   train_loss = 3.022
    Epoch  73 Batch   31/33   train_loss = 2.949
    Epoch  74 Batch    8/33   train_loss = 2.931
    Epoch  74 Batch   18/33   train_loss = 2.964
    Epoch  74 Batch   28/33   train_loss = 3.003
    Epoch  75 Batch    5/33   train_loss = 3.017
    Epoch  75 Batch   15/33   train_loss = 2.919
    Epoch  75 Batch   25/33   train_loss = 2.927
    Epoch  76 Batch    2/33   train_loss = 2.861
    Epoch  76 Batch   12/33   train_loss = 2.924
    Epoch  76 Batch   22/33   train_loss = 2.875
    Epoch  76 Batch   32/33   train_loss = 2.892
    Epoch  77 Batch    9/33   train_loss = 2.877
    Epoch  77 Batch   19/33   train_loss = 2.819
    Epoch  77 Batch   29/33   train_loss = 2.841
    Epoch  78 Batch    6/33   train_loss = 2.832
    Epoch  78 Batch   16/33   train_loss = 2.849
    Epoch  78 Batch   26/33   train_loss = 2.820
    Epoch  79 Batch    3/33   train_loss = 2.767
    Epoch  79 Batch   13/33   train_loss = 2.860
    Epoch  79 Batch   23/33   train_loss = 2.844
    Epoch  80 Batch    0/33   train_loss = 2.812
    Epoch  80 Batch   10/33   train_loss = 2.824
    Epoch  80 Batch   20/33   train_loss = 2.783
    Epoch  80 Batch   30/33   train_loss = 2.825
    Epoch  81 Batch    7/33   train_loss = 2.790
    Epoch  81 Batch   17/33   train_loss = 2.847
    Epoch  81 Batch   27/33   train_loss = 2.831
    Epoch  82 Batch    4/33   train_loss = 2.725
    Epoch  82 Batch   14/33   train_loss = 2.753
    Epoch  82 Batch   24/33   train_loss = 2.749
    Epoch  83 Batch    1/33   train_loss = 2.650
    Epoch  83 Batch   11/33   train_loss = 2.680
    Epoch  83 Batch   21/33   train_loss = 2.715
    Epoch  83 Batch   31/33   train_loss = 2.705
    Epoch  84 Batch    8/33   train_loss = 2.659
    Epoch  84 Batch   18/33   train_loss = 2.704
    Epoch  84 Batch   28/33   train_loss = 2.704
    Epoch  85 Batch    5/33   train_loss = 2.746
    Epoch  85 Batch   15/33   train_loss = 2.703
    Epoch  85 Batch   25/33   train_loss = 2.673
    Epoch  86 Batch    2/33   train_loss = 2.622
    Epoch  86 Batch   12/33   train_loss = 2.655
    Epoch  86 Batch   22/33   train_loss = 2.636
    Epoch  86 Batch   32/33   train_loss = 2.627
    Epoch  87 Batch    9/33   train_loss = 2.644
    Epoch  87 Batch   19/33   train_loss = 2.566
    Epoch  87 Batch   29/33   train_loss = 2.581
    Epoch  88 Batch    6/33   train_loss = 2.597
    Epoch  88 Batch   16/33   train_loss = 2.618
    Epoch  88 Batch   26/33   train_loss = 2.668
    Epoch  89 Batch    3/33   train_loss = 2.657
    Epoch  89 Batch   13/33   train_loss = 2.703
    Epoch  89 Batch   23/33   train_loss = 2.641
    Epoch  90 Batch    0/33   train_loss = 2.622
    Epoch  90 Batch   10/33   train_loss = 2.646
    Epoch  90 Batch   20/33   train_loss = 2.618
    Epoch  90 Batch   30/33   train_loss = 2.682
    Epoch  91 Batch    7/33   train_loss = 2.639
    Epoch  91 Batch   17/33   train_loss = 2.695
    Epoch  91 Batch   27/33   train_loss = 2.683
    Epoch  92 Batch    4/33   train_loss = 2.538
    Epoch  92 Batch   14/33   train_loss = 2.544
    Epoch  92 Batch   24/33   train_loss = 2.578
    Epoch  93 Batch    1/33   train_loss = 2.470
    Epoch  93 Batch   11/33   train_loss = 2.481
    Epoch  93 Batch   21/33   train_loss = 2.541
    Epoch  93 Batch   31/33   train_loss = 2.539
    Epoch  94 Batch    8/33   train_loss = 2.539
    Epoch  94 Batch   18/33   train_loss = 2.543
    Epoch  94 Batch   28/33   train_loss = 2.533
    Epoch  95 Batch    5/33   train_loss = 2.526
    Epoch  95 Batch   15/33   train_loss = 2.483
    Epoch  95 Batch   25/33   train_loss = 2.491
    Epoch  96 Batch    2/33   train_loss = 2.455
    Epoch  96 Batch   12/33   train_loss = 2.490
    Epoch  96 Batch   22/33   train_loss = 2.418
    Epoch  96 Batch   32/33   train_loss = 2.419
    Epoch  97 Batch    9/33   train_loss = 2.427
    Epoch  97 Batch   19/33   train_loss = 2.328
    Epoch  97 Batch   29/33   train_loss = 2.335
    Epoch  98 Batch    6/33   train_loss = 2.370
    Epoch  98 Batch   16/33   train_loss = 2.355
    Epoch  98 Batch   26/33   train_loss = 2.348
    Epoch  99 Batch    3/33   train_loss = 2.304
    Epoch  99 Batch   13/33   train_loss = 2.404
    Epoch  99 Batch   23/33   train_loss = 2.354
    Model Trained and Saved
    


```python
t1_train = time.time()

print(t1_train - t0_train, 'Seconds to train the nural network.')
```

    8603.382199048996 Seconds to train the nural network.
    

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    InputTensor = loaded_graph.get_tensor_by_name('input:0')
    InitialStateTensor = loaded_graph.get_tensor_by_name('initial_state:0')
    FinalStateTensor = loaded_graph.get_tensor_by_name('final_state:0')
    ProbsTensor = loaded_graph.get_tensor_by_name('probs:0')
    
    return (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed
    

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    idx_next_word = np.argmax(probabilities)
    next_word = int_to_vocab[idx_next_word]
    
    return next_word


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed
    

## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    moe_szyslak: hey, hey, hey, you got the lot of sex-- i got me too happen?
    homer_simpson:(laughing) i don't know you to don't the money that awesome.
    moe_szyslak:(excited) unlike all the manager?(whining) yeah, moe, the love of a bad thing. it was like a delightfully poulet au lodge?
    moe_szyslak: yeah, that's the worst thing that, homer. you got a multi-national job? i got a plan.(hammy noise)
    moe_szyslak: you know me? there, homer. i'm firing to be a job? i got a lot of sex in a darkest girl?
    moe_szyslak: don't be a crunch, celeste. you got for the way and the tremendous that a english thing it was?
    moe_szyslak: oh, you know, i got be else worse me. and you just got a" fix his life.
    homer_simpson:(to himself) no, you just know that works?
    moe_szyslak: oh, you got so one way on the wheel man.
    
    

# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckily there's more data!  As we mentioned in the beggining of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
