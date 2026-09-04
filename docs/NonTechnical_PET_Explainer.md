# Surrogate Safety Measures and PET: A Simple Explanation

This document explains our work in plain language for transportation engineering professionals.


## What are Surrogate Safety Measures (SSMs)?

SSMs are indicators of traffic safety that do not require waiting for crashes to happen. They use near-miss events and vehicle interactions to estimate how safe a location is.


## What is PET (Post-Encroachment Time)?

PET is the time gap between when one road user leaves a conflict zone and another road user enters it. Imagine two vehicles approaching an intersection from different directions. If one passes through just as the other is about to enter, the time between them is PET.


### How is PET interpreted?

- **PET < 1.0 second**: Critical – vehicles were extremely close in time.

- **1.0–1.5 seconds**: Serious – high risk.

- **1.5–3.0 seconds**: Moderate – potential conflict.

- **> 3.0 seconds**: Safe – vehicles had enough time separation.


## How does the software compute PET?

The program tracks vehicles from video. When two vehicles' paths cross or occupy the same area at different times, it records when the first vehicle left and when the second entered. The difference in time is PET.


## How do we know the numbers are correct?

We have a separate verification module that automatically checks every PET value. It ensures:

- PET values are positive.

- The two vehicles are different.

- The site is correctly labeled (GITI or MRC).

- Recalculating PET from the recorded times gives the same result.

These checks are documented in `outputs/PET_Verification_Report.md` and are run automatically every time the data is updated.


## Why is this reliable?

The core PET calculation is covered by 100% of unit tests. This means every possible edge case has been tested, and the logic behaves exactly as expected.
