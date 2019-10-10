/**
 * If you wish to use this implementation in anyway, please use the following citations
 * 
 * 1. Pillay, Bradley J., and Absalom E. Ezugwu. "Stock Price Forecasting Using Symbiotic Organisms Search Trained Neural Networks." In International Conference on Computational Science and Its Applications, pp. 673-688. Springer, Cham, 2019.
 * 2. Pillay, Bradley J., and Absalom E. Ezugwu. "On the performance of metaheuristics-trained neural networks for improved stock price prediction." article submitted to Neurocomputing - Journal, Elsevier.
 */
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StockPricePrediction
{
    class PSO_NN
    {
        int dim;
        int numParticles;
        int maxEpochs = 1000;

        // value bounds
        double minX = 1;
        double maxX = -1;
        double[] bestSolution;
        double[][] X;
        double[][] Y;

        NeuralNetwork nn;
        public PSO_NN(int dim, int numParticles, int numIterations, NeuralNetwork nn, double[][] X, double[][] Y)
        {
            this.dim = dim;
            this.numParticles = numParticles;
            this.maxEpochs = numIterations;
            this.X = X;
            this.Y = Y;
            this.nn = nn;
            bestSolution = nn.GetWeights();


        }

        public double CalculateFitnessValue(double[] individual)
        {
            nn.UpdateWeights(individual);
            return nn.Evaluate(X, Y).Average();
        }

        public void Train()
        {
            Random r = new Random();

            Particle[] swarm = new Particle[numParticles];
            double[] bestGlobalPosition = nn.GetWeights(); // best solution found by any particle in the swarm
            double bestFitness = nn.Evaluate(X, Y).Average();

            double[] tempPos = bestGlobalPosition;
            double error = bestFitness;
            double[] randomVelocity = new double[dim];

            for (int j = 0; j < randomVelocity.Length; ++j)
            {
                double lo = minX * 0.1;
                double hi = maxX * 0.1;
                randomVelocity[j] = (hi - lo) * r.NextDouble() + lo;
            }
            swarm[0] = new Particle(tempPos, error, randomVelocity, tempPos, error);

            if (swarm[0].error < bestFitness)
            {
                bestFitness = swarm[0].error;
                swarm[0].position.CopyTo(bestGlobalPosition, 0);
            }
            nn.UpdateWeights(bestGlobalPosition);

            double[] tempWeights = nn.GetWeights();
            for (int i = 1; i < numParticles; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    tempPos[j] = r.NextDouble();
                }

                nn.UpdateWeights(tempPos);
                error = nn.Evaluate(X, Y).Average();

                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = minX * 0.1;
                    double hi = maxX * 0.1;
                    randomVelocity[j] = (hi - lo) * r.NextDouble() + lo;
                }
                swarm[i] = new Particle(tempPos, error, randomVelocity, tempPos, error);

            }
            nn.UpdateWeights(tempWeights);

            double w = 0.9;
            double c1 = 2; // cognitive/local weight
            double c2 = 2; // social/global weight
            double r1, r2; // cognitive and social randomizations
            double probDeath = 0.01;
            int epoch = 0;

            double[] newVelocity = new double[dim];
            double[] newPosition = new double[dim];
            double newError;

            // main loop
            while (epoch < maxEpochs)
            {
                for (int i = 0; i < swarm.Length; ++i) // each Particle
                {
                    Particle currP = swarm[i]; // for clarity

                    // new velocity
                    for (int j = 0; j < currP.velocity.Length; ++j) // each component of the velocity
                    {
                        r1 = r.NextDouble();
                        r2 = r.NextDouble();

                        newVelocity[j] = (w * currP.velocity[j]) +
                          (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                          (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));
                    }
                    newVelocity.CopyTo(currP.velocity, 0);

                    // new position
                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];
                        if (newPosition[j] < minX)
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }
                    newPosition.CopyTo(currP.position, 0);

                    tempWeights = nn.GetWeights();
                    nn.UpdateWeights(newPosition);
                    newError = nn.Evaluate(X, Y).Average();
                    currP.error = newError;

                    if (newError < currP.bestError)
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }

                    if (newError < bestFitness)
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestFitness = newError;
                    }
                    else
                    {
                        nn.UpdateWeights(tempWeights);
                    }

                    // death? maybe
                    double die = r.NextDouble();
                    if (die < probDeath)
                    {
                        // new position, leave velocity, update error
                        for (int j = 0; j < currP.position.Length; ++j)
                            currP.position[j] = (maxX - minX) * r.NextDouble() + minX;

                        tempWeights = nn.GetWeights();
                        nn.UpdateWeights(currP.position);
                        currP.error = nn.Evaluate(X, Y).Average();

                        currP.position.CopyTo(currP.bestPosition, 0);
                        currP.bestError = currP.error;

                        if (currP.error < bestFitness) // global best by chance?
                        {
                            bestFitness = currP.error;
                            currP.position.CopyTo(bestGlobalPosition, 0);
                        }
                        else
                        {
                            nn.UpdateWeights(tempWeights);
                        }
                    }

                }
                ++epoch;
            }

            bestGlobalPosition.CopyTo(bestSolution, 0);
        }

        public double RMSE(double[] PredictedOutput, double[] ExpectedOutput)
        {
            double rmse = 0;
            for (int j = 0; j < PredictedOutput.Length; j++)
            {
                rmse += Math.Pow((ExpectedOutput[j] - PredictedOutput[j]), 2);
            }
            rmse = rmse / PredictedOutput.Length;
            rmse = Math.Sqrt(rmse);

            return rmse;
        }

        public double[] GetBestSolution()
        {
            return bestSolution;
        }

        public NeuralNetwork GetNeuralNetwork()
        {
            return nn;
        }

        public double GetBestError()
        {
            nn.UpdateWeights(bestSolution);
            return nn.Evaluate(X, Y).Average();
        }

        public void UpdateNeuralNetworkWithBestWeights()
        {
            nn.UpdateWeights(bestSolution);
        }
    }
}
