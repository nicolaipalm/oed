x: one single design x0: array of designs x y: output

supposed to grow

---
Write with cells the example script in here (benchmarking and dashboard)
and give reference to notebook

---
You can define your own metric (section...-> )

---
In dashboard: learn how to interpret the plots , email for questions, view on github -> plot also the functions
dependant of parameters ...
---
!!! Web application saved as html -> freeze results for paper!!!! -> this is the use of dashboard What i need:

- paper ref
- author
- plots
- meta data ?
- save data as csv

---
ToDO:

- in aging model nau werte über 10000 (bei parametern zwischen 0.001 und X) fehler?? Siehe Print in aging model
- docu-> coding beispiel für jede metric: max der parameter und andere error funktion für integral -> erkläre wieso
  metric mit max zwei plotting
- HOW TO DIRECTORY docu neues (super scvhlechtes, und custom) design (max mit geschätzem theta...individuell...), neue metrik, daten speichern,
- setup git -> also nb extensions...
- (neues projekt mit dashboard)
- keine problems
- benchmark whole process
- designs need to be saved for benchmarking results 
- why so much slower than other module?
- test if parameters are redundant -> in validation error we see almost no difference ... cross validation has errors in
  it
- -> !!! cross validation: comparing the values is difficult because the DESIGN differ (think of linear function: if
  designs are great than error will be great...BUT: we compare different models on SAME designs... not on different
  designs. AND: here LH has an optimal property since it allows cross validation to better estimate the integral of
  erreor-> not needed to be implemented in this library...?
- (pi design needs an initial design within-> otherwise determinant not invertible ...)

---

- statistical model-> in context of FI (assume there exists partial derivative of likelihood etc)
- MLE on statistical model
- we are interested how different designs of experiments perform
- design where the ucertainty in the experiments has minimum effect on the parameter estimation 

Title:
optimal experimental desgin for parameter individual estimators of/in parametric models with gaussian white noise
