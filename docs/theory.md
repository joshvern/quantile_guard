# Theory (very short)

## Quantile regression as a linear program

Quantile regression estimates conditional quantiles by minimizing the *check loss*.
With slack variables for positive/negative residuals, the objective becomes linear and the problem can be written as a linear program (LP).

When fitting multiple quantiles jointly, additional linear constraints can be added to encourage *non-crossing* quantiles on the training data.

## Why PDLP?

This package uses the PDLP solver from Google OR-Tools to solve the LP.
PDLP is a first-order method designed for large-scale linear programs, which can be a good fit for optimization-formulated regression problems.

(For details on the exact LP used, see the source and the math section in the project README.)
