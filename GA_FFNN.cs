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
    class GA_FFNN
    {
        private double[][] population;
        private double[][] X;
        private double[][] Y;
        private double[] bestIndividual;
        private double bestFitness;

        private int popSize;
        private double[] fitness;
        private NeuralNetwork nn;
        private Random r;

        public GA_FFNN(int popSize, double[][] X, double[][]Y)
        {
            this.popSize = popSize;
            nn = new NeuralNetwork(2, 2, 8);
            population = new double[popSize][];
            fitness = new double[popSize];
            r = new Random();
            this.X = X;
            this.Y = Y;

            for (int i = 0; i < popSize; i++)
            {
                population[i] = new double[(2 * 2 * 8) + 2];
                for (int j = 0; j < (2 * 2 * 8) + 2; j++)
                {
                    population[i][j] = r.NextDouble();
                }

                nn.UpdateWeights(population[i]);
                fitness[i] = nn.Evaluate(X, Y).Average();
            }

            ChooseBestIndividual();
        }

        public void EvolvePopulation()
        {
            List<double[]> newPop = new List<double[]>();
            newPop.Add(bestIndividual);

            for (int i = 1; i < popSize; i++)
            {
                double[] parentA = RouletteSelection();
                double[] parentB = RouletteSelection();

                double[] crossover = Crossover(parentA, parentB);
                double[] child = Mutation(crossover);

                newPop.Add(child);
            }

            population = newPop.ToArray();
            for (int i = 0; i < popSize; i++)
            {
                nn.UpdateWeights(population[i]);
                fitness[i] = nn.Evaluate(X, Y).Average();
            }

            ChooseBestIndividual();
        }

        public double[] Crossover(double[] IndividualA, double[] IndividualB)
        {
            double[] kid = new double[IndividualA.Length];
            for (int i = 0; i < kid.Length; i++)
            {
                kid[i] = r.NextDouble() <= 0.5 ? IndividualA[i] : IndividualB[i];
            }

            return kid;
        }

        public double[] Mutation(double[] Individual)
        {
            double mutationRate = 1.0 / Individual.Length;
            for (int i = 0; i < Individual.Length; i++)
            {
                if (r.NextDouble() <= mutationRate)
                {
                    Individual[i] = r.NextDouble();
                }
            }
            return Individual;
        }

        public double[] RouletteSelection()
        {
            double Sum = fitness.Sum();
            double[] wheel = new double[fitness.Length];
            double wheelSum = 0;
            for (int i = 0; i < wheel.Length; i++)
            {
                double temp = fitness[i] / Sum;
                wheelSum += temp;
                wheel[i] = wheelSum;
            }

            double randomValue = r.NextDouble();

            int index = 0;
            while (wheel[index] < randomValue)
            {
                index++;
            }

            return population[index];
        }

        public void ChooseBestIndividual()
        {
            int minIndex = 0;

            for (int i = 1; i < fitness.Length; i++)
            {
                if (fitness[i] < fitness[minIndex])
                {
                    minIndex = i;
                }
            }

            bestIndividual = population[minIndex];
            bestFitness = fitness[minIndex];
        }

        public double GetFitness(double[] individual)
        {
            nn.UpdateWeights(individual);
            return nn.Evaluate(X, Y).Average(); 
        }

        public double[] GetBestIndividual()
        {
            return bestIndividual;
        }
    }
}
