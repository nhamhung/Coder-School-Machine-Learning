# Implement a children doodle website based on Google `Quick, Draw!` and `Sketch-RNN`

[Link to Final Project](https://github.com/nhamhung/Final-Project-With-SketchRNN)

## Introduction

- At a younger age, children begin to learn about the world around them and express their perception through drawings. Interestingly, the super complex real world becomes greatly simplified under children's hands where they model everything as simple sketches. And through these sketches, children learn to extract distinct features from a complex object and turn them into comprehensible forms that represent the object well. In fact, for many among us, the very first sketches which we drew from young would also follow us way later into adulthood. It is a manifestation of our identity, culture, upbringings and more specifically, the country where we grew up. This idea is what interests and motivates me to begin looking into sketching as an inspiration for my final project.
- On a more personal note, I cannot draw very well. And so, it is surprisingly pleasant to come from the place of someone analysing and learning from others' drawings. Luckily, the **Google's `Quick, Draw!` dataset** has just what I need. A storage of more than **300 object classes**, each of which is well splitted into **70000 for training, 2500 for valiation and 2500 for testing**, this dataset serves well for my purpose of exploration. Moreover, in each sketch record, valuable information such as the author's country of origin also serves as a study point of how people from different countries draw the same object differently from one another. This dataset will thus be my main source of training and evaluating for my final product.
- On a quick note, for this dataset, each sketch in each object class such as `Cat` comes with the following variables:
    - **“word”** — the class label of that drawing
    - **“country code”** — country of origin of the drawer
    - **“timestamp”** — timestamp of the drawing
    - **“recognized”** — indication of the app’s successful prediction
    - **“drawing”** — stroke-base data specific for doodle images; each drawing is made up of multiple strokes in the form of matrices

## Final product

- Upon more research about the `Quick, Draw!` dataset, I came across David Ha's paper on his implementation of a model called `Sketch-RNN`. This model is a *"Sequence-to-Sequence Variational Autoencoder"*, with the encoder being a "*bidirectional RNN that takes in a sketch as an input and output a latent vector z*." This latent vector is not a deterministic output but a random vector conditioned on the input sketch. The decoder is an "*autoregressive RNN that samples output sketches conditional on a given latent vector z*". 

- With this model, I aim to create a **Children doodling website** that autocompletes your sketch to support the drawing process. My main target audience are children who are learning to draw or learning about different animals and objects in daily life. Of course, adults are also suitable to use the website because for a single object can have so many variations and this enriches our own creativity. I will also divide this project into different iterations with features that build incrementally upon the previous.

## Iterations

### 1. Autocomplete your sketch
- Autocomplete your current sketch based on a selected `object` on a single canvas.

### 2. Multiple-variation autocomplete
- Autocomplete your current sketch based on a selected `object` but display multiple variations of the completed sketch on multiple side canvases.

### 3. Draw alongside with you
- As you draw a selected `object`, hint the next stroke to help you complete that object more accurately.

### 4. Add musical ML models to generate music while drawing
- Enhance the drawing experience with machine-generated musical elements. Relax while you sketch.

## Summary of `Teaching Machines to Draw`:

- Instead of working with the Quickdraw dataset in the form of "raster images" represented as a 2D grid of pixels, I want to explore David Ha's approach of lower-dimensional vector-based representation inspired by the way people draw. The sketch-rnn model is based on the "sequence-to-sequence" autoencoder framework. It incorporates "variational inference" and utilizes "hypernetworks" as recurrent neural network cells. The goal of a seq2seq autoencoder is to train a network to encode an input sequence into a vector of floating point numbers, called a *latent* vector. From this latent vector, he reconstructs an output sequence using a decoder that replicates the input sequence as closely as possible.

![](https://i.imgur.com/Kkn4sm1.png)

- In the model, David Ha deliberately adds noise to the latent vector. By inducing noise into the communication channel between the encoder and the decoder, the model is no longer be able to reproduce the input sketch exactly, but instead must learn to capture the essence of the sketch as a noisy latent vector. The decoder takes this latent vector and produces a sequence of motor actions used to construct a new sketch.

- Also, he explains that the reconstructed sketches are not copies of input sketches, but are instead new sketches with similar characteristics as the input. He shows that latent vectors indeed encode conceptual features of a sketch.

![](https://i.imgur.com/INPOYt3.png)

- He finds that sketch drawing analogies are possible for model trained on both cat and pig sketches. For example, one can subtract the latent vector of an encoded pig head from the latent vector of a full pig, to arrive at a vector that represents the concept of a body. Adding this difference to the latent vector of a cat head results in a full cat (i.e. cat head + body = full cat). These drawing analogies allow us to explore how the model organizes its latent space to represent different concepts in the manifold of generated sketches.

- Decoder module can be used as a standalone model and trained to predict different possible endings of incomplete sketches. 

![](https://i.imgur.com/Ifv2R3b.png)

- Or taking it further, have different models complete the same incomplete sketch.

![](https://i.imgur.com/BPAA9MG.png)

## Summary of `A Neural Representation of Sketch Drawings`

- Goal: Train machines to draw and generalize abstract concepts in a manner similar to humans. From young, we have developed the ability to communicate what we see through pen and paper, by expressing a sequential, vector representation of an image in a short sequence of strokes. Similarly, we present to the machines each sketch as a sequene of motor actions controlling a pen: which direction to move, when to lift the pen up and when to stop drawing.

- Overview of the dataset:
    - Each class of QuickDraw contains 70K training samples, 2.5K validation and 2.5K test.
    - Data format represents a sketch as a set of pen stroke actions. The initial absolute coordinate is located at the origin. A sketch is a list of points, and each point is a vector consisting of 5 elements: $$(\Delta x,\Delta y, p1, p2, p3)$$
    - The first two elements are offset distance in the x and y directions of the pen from the previous point.
    - The last 3 elements represents a binary one-hot vector of 3 possible states. $$p1$$ means the pen is currently touching the paper and a line will be drawn connecting the next point to the current point. $$p2$$ means the pen will be lifted from the paper after the current point and no line will be drawn next. $$p3$$ indicates that the drawing has ended and subsequent points including the current point will not be rendered.

- **Sequence-to-sequence Variational Autoencoder (VAE)**

    ![](https://i.imgur.com/Kkn4sm1.png)

    - Encoder: A bidirectional RNN that takes a sketch as an input and outputs a latent vector z.
    - We feed in a sketch sequence S and the same sketch sequence in reverse order S-reverse to obtain the 2 h's.    

    ![](https://i.imgur.com/eXkt0rg.png)
    
    - Then construct a latent vector z:

    ![](https://i.imgur.com/vB1B4AU.png)

    - The resulting latent vector z is non-deterministic output and a random vector conditional on the input sketch.

    - Decoder: autoregressive RNN that samples output sketches conditional on a given latent vector z.

    - At each step i of the decoder RNN, we feed the previous point, $S_{i−1}$ and the latent vector z in as a concatenated input $x_i$, where $S_0$ is defined as (0, 0, 1, 0, 0). The output at each time step are the parameters for a probability distribution of the next data point $S_i$.

    ![](https://i.imgur.com/BsSIcaf.png) 
    ![](https://i.imgur.com/aJqjUe2.png)
    ![](https://i.imgur.com/JjgpQ90.png)
    ![](https://i.imgur.com/K6cE07V.png)
    ![](https://i.imgur.com/vATR23E.png)

    - All sequences are generated to a length of $N_{max}$ which is the length of the longest sketch in training dataset. N_{max} is also a hyperparameter.
    - After training, sample sketches from the model. During sampling process, generate parameters for both GMM and categorical distribution at each time step and sample an outcome $S'_i$ as input for the next time step. Continue to sample until $p_3 = 1$ when we reach $i=N_{max}$. The sampled output is not deterministic, similar to the encoder, and conditioned on the input latent vector z.
    - Can control the level of randomness of the samples during the sampling process by a temperature parameter $T$:
    ![](https://i.imgur.com/uVztK9H.png)
    - $T$ is set between 0 and 1. When $T$ approaches 0, the model becomes deterministic and samples will consist of the most likely point in the probability density function. 

## Steps I took:

### 1. Try out the dataset in 2D format with a multi-classification task based on CNN model

- Making a simple Flask App classification website based on your drawing on a canvas.
    - Only 6 classes involved: `cat`, `octopus`, `bat`, `giraffe`, `camel` and `sheep`
    - For each class, only train on 10000 samples and the accuracy reached 93%
    - What I noticed: Overall, the model picked up dominant features of each animal such as the `cat's ears, octopus' legs, bat's wings, giraffe's neck, camel's back and sheep's body`

- (Here is the notebook for this task)[]

### 2. Study the model code implementation:

#### a. `dataset.py`

- `cleanup(d, limit=1000)` method: convert data to float and apply a `clip(-1000, 1000)` where values below and above limit will become -1000 and 1000 respectively.
- `calc_scale_factor(data)` and `normalize(data,scale_factor)` is to normalize the data
- `pad(data, seq_len)` to add padding
- `augment(data, prob)`, `random_scale(data, eps)` and `pad(data, seq_len)` are used to `preprocess(d, seq_len, eps, prob)` data
- `data_gen(data, scale_factor)` 

#### b. `models.py`

##### `models.SketchRNN(hps)` will construct a sketchrnn model
- Encoder: 
    - Input Layer: shape=(131,5) as max_seq_len=131
    - Encoder LSTM Cell: with enc_rnn_size=256 and recurrent_dropout_prob=0.1
    - Encoder Output: A Bidirectional RNN layer that takes in Input Layer
    - mu = Dense Layer taking in Encoder Output
    - sigma = Dense Layer taking in Encoder Output
    - Latent_z = Lambda Layer that applies `reparameterize()` to mu and sigma
    - Overall: Model(inputs=encoder_input, outputs=[latent_z, mu, sigma], name="encoder")
- Initial_state:
    - z_input: Input Layer(shape=(z_size=128,), name="z_input")
    - initial_state = Dense Layer(units=dec_rnn_size * 2, activation="tanh", kernel_initializer=K.initializers.RandomNormal(mean=0.0, stddev=0.001)) that takes in z_input
    - states = tf.split(initial_state, 2, 1) where tf.split(X, row = n, column = m) is used to split the data set of the variable into n number of pieces row wise and m numbers of pieces column wise
    - Overall: Model(inputs=z_input, outputs=states, name="initial_state")
- Decoder:
    - Input Layer: shape(None, 5)
    - z_input: Input(shape=(z_size=128,))
    - initial_h_input = Input Layer(shape=(dec_rnn_size=512,), name="init_h")
    - initial_c_input = Input Layer(shape=(dec_rnn_size=512,), name="init_c")
    - Decoder LSTM = LSTM Layer(units=dec_rnn_size=512, recurrent_dropout=recurrent_dropout_prob=0.1)
    - tile_z = tf.tile(tf.expand_dims(z_input, 1), [1, tf.shape(decoder_input)[1], 1]) where tf.tile() new tensor by replicating input multiples times. The output tensor's i'th dimension has input.dims(i) * multiples[i] elements
    - decoder_full_input = tf.concat([decoder_input, tile_z], -1)
    - decoder_output, cell_h, cell_c = Decoder LSTM(decoder_full_input, initial_state=[initial_h_input, initial_c_input])
    - output_layer = Dense Layer(units=num_mixture=20 * 6 + 3, name="output")
    - output = output_layer(decoder_output)
    - Overall: Model(inputs=[decoder_input, z_input, initial_h_input, initial_c_input],
                     outputs=[output, cell_h, cell_c],  name="decoder",)
- Model Builder:
    - Encoder produces latent vector: z_out, mu value: mu_out, sigma value: sigma_out
    - Initializer use latent vector to produce init_h, init_c
    - Decoder produces some output, cell_h, cell_c
    - Overall, model takes in [encoder_input, decoder_input] and produces [output, mu_out, sigma_out]

##### Other methods:

- `self.load_weights(path)` to load weights from a weight file
- `self.sample(temperature=1.0, greedy=False, z=None)`: If z is None, meaning there is no latent vector, generate a random latent vector with standard normal distribution using np.random.rand(1, z_size=128). Then, only the decoder part is used to generate `strokes` with length = `max_len`. To generate strokes, at each iteration until `i = max_len`, `get_mixture_coef()` is used on `output` from decoder to generate `o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen`. Then, `sample_gaussian_2d()` is used to generate `next_x1` and `next_x2`. `next_x1` and `next_x2` are used to generate `strokes[i]` until `max_len`.

#### c. `utils.py`


## New Terminology:

### `NDJSON`: 
- Newline delimited JSON. NDJSON is a convenient format for storing or streaming structured data that may be processed one record at a time. It works well with unix-style text processing tools and shell pipelines. It's a great format for log files. It's also a flexible format for passing messages between cooperating processes.

- Example:
```
 {"some":"thing"}
 {"foo":17,"bar":false,"quux":true}
 {"may":{"include":"nested","objects":["and","arrays"]}}
```
(with `\n` as line separators)

### `.npz` file
- Save several arrays into a single file in uncompressed `.npz` format.
- Stand for numpy.savez
- To open, use 
```python
data = np.load('/content/SketchRNN_tf2/data/cat.npz',encoding='latin1',allow_pickle=True)
```

### `np.newaxis`
- Increase dimension of existing array 

### Latent vector

### Autoregressive RNN

### Gaussian mixture model (GMM)

### Categorical distribution

## Timeline:

[My Timeline Proposal](https://docs.google.com/spreadsheets/d/17G5fKtnfsDqerNZWGlwXNfvSwzrvNrx-CG722RpRikA/edit#gid=0)

## References:

- David Ha: 
[Teaching Machines to Draw](https://ai.googleblog.com/2017/04/teaching-machines-to-draw.html) and [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)


