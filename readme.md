# Implicit Discourse Parsing

## Intro

This is the course project for *Natural Language Processing* in Spring, 2016. The course instructor is Hai, Zhao.

Discourse parsing is a task trying to predict the relation between two neighbouring arguments:

> Arg1: This should be a sample.
> Arg2: (*However*) I'm too lazy to find a real one.

The example above shows two arguments, if our system performs correctly, it will output "Contrast" as the output based on the understanding of these arguments. Note that there are 11 potential relations we need to predict. 

The task is "implicit" because we cannot find an explicit word/phrase within the argument that can well represent the relation. However, we can insert a connective word/phrase between them without losing the fluency. In the example above, we can use "However" as the connective word. 

## Round 1

I'll first try using a pre-constructed CNN stucture. In order to turn the arugments into input matrix, my first thought is to contacanate pre-trained vectors, with several *paddings* between the arguments.

Let's see how it works :) 