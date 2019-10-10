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
    class SOS_NN
    {
        int populationSize;
        int maxIterations;
        int currentIteration;

        double[][] population;
        double[] fitnessValues;
        int indexOfBestSolution;
        double[] bestSolution;
        double[][] X;
        double[][] Y;

        NeuralNetwork nn;

        public SOS_NN(int maxIterations, int populationSize, NeuralNetwork nn, double[][] X, double[][] Y)
        {
            this.maxIterations = maxIterations;
            this.populationSize = populationSize;
            this.X = X;
            this.Y = Y;
            this.nn = nn;
            this.bestSolution = nn.GetWeights();

            fitnessValues = new double[populationSize];

            Random r = new Random();
            population = new double[populationSize][];
            population[0] = bestSolution;
            indexOfBestSolution = 0;
            for (int i = 1; i < populationSize; i++)
            {
                population[i] = new double[bestSolution.Length];
                for (int j = 0; j < population[i].Length; j++)
                {
                    population[i][j] = r.NextDouble();
                }
            }

            double[] tempWeights = nn.GetWeights();
            for (int i = 0; i < populationSize; i++)
            {
                fitnessValues[i] = CalculateFitnessValue(population[i]);
            }
            nn.UpdateWeights(tempWeights);
        }

        public double CalculateFitnessValue(double[] individual)
        {
            nn.UpdateWeights(individual);
            return nn.Evaluate(X, Y).Average();
        }

        public void Train()
        {
            Random r = new Random();
            for (int i = 0; i < maxIterations; i++)
            {
                currentIteration = 0;
                do
                {
                    currentIteration++;
                    // mutualism
                    Mutualism(r.Next(0, populationSize));

                    // commensalism
                    Commensalism(r.Next(0, populationSize));
                    // parasitism
                    Parasitism(r.Next(0, populationSize));

                    // choose best individual
                    ChooseBestIndividual();
                } while (!(currentIteration < populationSize));
            }
        }

        public void Mutualism(int r)
        {
            double[] randomIndividual = new double[bestSolution.Length];
            Array.Copy(population[r], randomIndividual, randomIndividual.Length);

            double[] mutualIndividual = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                double d = population[currentIteration][i];
                mutualIndividual[i] = (population[currentIteration][i] + randomIndividual[i]) / 2;
            }
            Random random = new Random();
            int benefitFactorOne = random.Next(1, 3);
            int benefitFactorTwo = random.Next(1, 3);

            double[] Individual_INew = new double[bestSolution.Length];
            double[] Individual_JNew = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                Individual_INew[i] = (population[currentIteration][i] + random.NextDouble() * (bestSolution[i] - mutualIndividual[i] * benefitFactorOne));
                Individual_JNew[i] = (randomIndividual[i] + random.NextDouble() * (bestSolution[i] - mutualIndividual[i] * benefitFactorTwo));
            }

            double[] currentWeights = nn.GetWeights();
            nn.UpdateWeights(Individual_INew);
            double fitnessA = nn.Evaluate(X, Y).Average();

            nn.UpdateWeights(population[currentIteration]);
            double fitnessB = nn.Evaluate(X, Y).Average();

            if (fitnessA <= fitnessB)
            {
                for (int i = 0; i < bestSolution.Length; i++)
                {
                    population[currentIteration][i] = Individual_INew[i];
                }
                fitnessValues[currentIteration] = fitnessA;
            }

            nn.UpdateWeights(Individual_JNew);
            fitnessA = nn.Evaluate(X, Y).Average();

            nn.UpdateWeights(population[r]);
            fitnessB = nn.Evaluate(X, Y).Average();

            if (fitnessA <= fitnessB)
            {
                for (int i = 0; i < bestSolution.Length; i++)
                {
                    population[r][i] = Individual_JNew[i];
                }
                fitnessValues[r] = fitnessA;
            }

            nn.UpdateWeights(currentWeights);
        }

        public void Commensalism(int r)
        {
            double[] randomIndividual = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                randomIndividual[i] = population[r][i];
            }

            Random random = new Random();
            double[] Individual_INew = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                Individual_INew[i] = (population[currentIteration][i] + random.Next(-1, 2) * (bestSolution[i] - population[r][i]));
            }

            double[] tempWeights = nn.GetWeights();
            nn.UpdateWeights(Individual_INew);
            double fitnessA = nn.Evaluate(X, Y).Average();

            nn.UpdateWeights(population[currentIteration]);
            double fitnessB = nn.Evaluate(X, Y).Average();

            if (fitnessA <= fitnessB)
            {
                for (int i = 0; i < bestSolution.Length; i++)
                {
                    population[currentIteration][i] = Individual_INew[i];
                }
                fitnessValues[currentIteration] = fitnessA;
            }
            nn.UpdateWeights(tempWeights);
        }

        public void Parasitism(int r)
        {
            double[] randomIndividual = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                randomIndividual[i] = population[r][i];
            }

            double[] parasite = new double[bestSolution.Length];
            for (int i = 0; i < bestSolution.Length; i++)
            {
                parasite[i] = population[currentIteration][i];
            }
            for (int i = 0; i < bestSolution.Length; i++)
            {
                int fromPopulation = new Random().Next(0, populationSize);
                parasite[i] = population[fromPopulation][i];
            }

            double[] tempWeights = nn.GetWeights();
            nn.UpdateWeights(parasite);
            double fitnessA = nn.Evaluate(X, Y).Average();

            nn.UpdateWeights(population[r]);
            double fitnessB = nn.Evaluate(X, Y).Average();

            if (fitnessA <= fitnessB)
            {
                for (int i = 0; i < bestSolution.Length; i++)
                {
                    population[r][i] = parasite[i];
                }
                fitnessValues[r] = fitnessA;
            }
            nn.UpdateWeights(tempWeights);
        }

        public void ChooseBestIndividual()
        {
            double maxValue = double.MaxValue;
            int maxIndex = -1;

            for (int i = 0; i < populationSize; i++)
            {
                if (fitnessValues[i] < maxValue)
                {
                    maxValue = fitnessValues[i];
                    maxIndex = i;
                }
            }

            for (int i = 0; i < bestSolution.Length; i++)
            {
                bestSolution[i] = population[maxIndex][i];
            }
            indexOfBestSolution = maxIndex;
            UpdateNeuralNetworkWithBestWeights();
        }

        public void UpdateNeuralNetworkWithBestWeights()
        {
            nn.UpdateWeights(bestSolution);
        }

        public NeuralNetwork GetNeuralNetwork()
        {
            return nn;
        }
    }
}
