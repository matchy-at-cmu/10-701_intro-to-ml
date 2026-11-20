#import "/template.typ": cheatsheet

#show: cheatsheet.with(
  title: [10-701 Introduction to Machine Learning],
  authors: "Matchy",
  paper-size: "us-letter"
)

#include "sections/01-fairness.typ"
#include "sections/02-embedding.typ"
#include "sections/03-cnn.typ"
#include "sections/04-rnn.typ"
#include "sections/05-transformer.typ"
#include "sections/06-large-models.typ"
#include "sections/07-reinforcement-learning.typ"
#include "sections/08-learning-theory.typ"
