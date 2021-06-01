#!/bin/bash

chmod +x scripts/*sh

echo Univariate G-K
./scripts/univariate_g-and-k.sh

echo G-K
./scripts/g-and-k.sh
./scripts/g-and-k_increase_n_sim.sh  # test performance of BSL and semiBSL with increasing number of simulations
./scripts/g-and-k_multiple_random_seeds.sh # test performance of BSL and semiBSL with different random seeds

echo Univariate Cauchy G-K
./scripts/univariate_Cauchy_g-and-k.sh

echo Multivariate Cauchy G-K
./scripts/Cauchy_g-and-k.sh

echo misspecified normal location
./scripts/normal_location_misspec.sh

echo MA2
./scripts/MA2.sh

echo MG1
./scripts/MG1.sh

