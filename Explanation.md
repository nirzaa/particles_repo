# Explanation

We will take our data and split it to train and test with a ratio of 0.8.
Therefore, we will have 180 events for testing.
Each event has different number of particles with different energies. We will split their energies into 20 bins representing energies between 0 and 13 GeV.
We will predict the number of particles in each bin for each event and plot the sum of all of the events, namely all of the particles that are in the events split into 20 bins of energies.
We will calculate the average number of particles per event, and also the total number of particles in all of the events together.
We will exhibit the output of the neural net and also the target values.