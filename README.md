# Posterior-Decoding-and-Viterbi
Posterior decoding and Viterbi algorithms on Hidden Markov Models to detect C/G rich regions in a DNA sequence.

## Problem description
DNA sequences are made up of the nucleotides adenine (A), guanine (G), cytosine (C), and thymine (T). In the genomes of organisms, these are not distributed randomly. Rather, some regions will have more A's and T's, while other regions will have more C's and G's. This is important to note because C/G rich regions tend to have more genes, while A/T rich regions have more non-functional DNA.

This program uses Hidden Markov Models (HMMs) to detect C/G rich regions. The program will use the Viterbi and Posterior algorithms.

## Output and Testing
The program will output two files in the following format:
* The first file is the output of the Viterbi algorithm. The first and second lines are the number of inferred 0 or 1 states respectively in the inferred sequence of states.
* The second file is similar to the first, but is generated using the Posterior Decoding Algorithm.

To run the program, enter the following command: ```python a3_template.py "small.out"```
