# NLP Notes

## N-grams

### Unigram probabilities

* Word probability `P(w_i)`: Frequency of word `w_i` divided by total number of words in corpus

```
P(w_i) = count( w_i ) / count ( total number of words)
```

### Bigram probabilities

* Probability that word `w_(i-1)` is followed by `w_i` (a.k. conditional probability)

```
P(w_i | w_(i-1) ) = count( w_(i-1), w_i) / count (w_(i-1))
```

## Mixed

* **Distributional Similarity**: Semantics of words meanings under the context they appear
* **Distributed Representation**: e.g. one-hot encoding or word vectors
* **Loss Function**
    * also known as **cost function**
    * Real number associated with event
    * Optimization problem tries to minimize the loss
* **Objective Function**
    * is a loss function or ist opposite (**reward function**)
    * Optimization problem tries to maximize this function
* **Softmax**-function
    * input: k-dimensional (arbitrarily) real valued vector
    * output: k-dimensional vector with each value in `(0,]`, and all entries sum up to `1`
    * Formula: `exp(a) / sum(exp(a))`
    * Can used as probability distributions over `k` outcomes (categories)
* **Word2Vec**
    * Initial objective: Maximize the probability of any context word given the current focus word
        1. multiply the conditional probabilities of focus with context word
        2. multiply all the focus word probabilities
        3. Objective `J'`: maximize this product
    * Real world: turn `J'` into `J` by using negative log likelihood (products become sums). The negative turns Maximize problem into minimization problem
* **Cross-entropy loss**
    * Usual loss function for prediction probabilities
    ```python
    def CrossEntropy(yHat, y):
        # yHat is predicted
        if yHat == 1:
            return -log(y)
        else:
            return -log(1 - y)
    ```
* **Hinge loss**
    * Used for classification
    ```python
    def Hinge(yHat, y):
        # yHat is predicted
        return np.max(0, 1 - yHat * y)
    ```