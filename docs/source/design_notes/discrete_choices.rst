.. _discrete_choices:

===============================
Thinking about discrete choices
===============================


Flat or nested representation
-----------------------------

- By flat representation I mean that there is only one discrete choice. An example would
  be the choice set
  ``[insured-working, uninsured-working, insured-not-working, uninsured-not-working]``
- By nested I mean that there is one discrete choice per discrete choice variable. An
  example would be [insured, uninsured], [working, not-working].
- Nested decision means that there are more choice variables, some of which might be
  "simple_variables". This could be a computation advantage.
- Nested decisions might also lead to more intuitive user written utility functinos
  (e.g. nonpecuniary reward of being insured is easy if there is an insured-dummy.

-> I think we want nested choices

Categorical or integer
----------------------

- By categorical I mean "insured": {"options": ["insured", "uninsured"]}
- By integer I mean "insured": {"options": [0, 1]}
- Internally, we will always store the variables as integers
- If the user specifieds categories and uses them in his functions, it is very likely
  that if conditions are required. For integers those can be replade by multiplications
  (especially with binary choices). I think people are good at using those tricks and
  would like to have the control to do so.
- Mixed forms where codes are used in utility functions but categories are specified
  to be used in the simulated dataset are possible and could easily be added later.

-> for now I would just choose an integer representation

Open Questions: Is there ever a real conceptual difference of flat vs. nested discrete
choices? Are there computational differences except for state space compression?
Do nested decisions make the iid assumption weaker by allowing for different scales
of the error across choice variables?


The Emax Calculation
--------------------

- In the idd extreme-value case the Emax calculation is not computationally difficult
- However, if discrete choices are complex_variables, i.e. if there are restrictions
  on the choice set, it requires the evaluation of a logsumexp (or similar function) on
  portions of an array axis, where the portion length is not constant.
- For numpy ufuncs, such a calculation can be expressed with ``reduceeat``, there is no
  jax equivalent for that, nor any version that works for arbitrary functions.
- Will probably have to use numba for now and make a feature request for jax.
