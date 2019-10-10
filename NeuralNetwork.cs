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
    class NeuralNetwork
    {
        private int numInputs;
        private int numOutputs;
        private int numHidden;

        // weights & bias
        private double[][] ihWeights; // input-hidden
        private double[][] hoWeights; // hidden-output
        private double[] bias; // bias

        // layers
        private double[] inputLayer;
        private double[] hiddenLayer;
        private double[] outputLayer;

        public NeuralNetwork(int numInputs, int numOutputs, int numHidden)
        {
            Random r = new Random();

            this.numHidden = numHidden;
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;

            this.ihWeights = new double[numInputs][];
            for (int i = 0; i < numInputs; i++)
            {
                this.ihWeights[i] = new double[numHidden];
                for (int j = 0; j < numHidden; j++)
                {
                    ihWeights[i][j] = r.NextDouble();
                }
            }

            this.hoWeights = new double[numHidden][];
            for (int i = 0; i < numHidden; i++)
            {
                hoWeights[i] = new double[numOutputs];
                for (int j = 0; j < numOutputs; j++)
                {
                    hoWeights[i][j] = r.NextDouble();
                }
            }

            this.bias = new double[2];
            for (int j = 0; j < this.ihWeights.Length; j++)
            {
                bias[j] = r.NextDouble();
            }

            this.inputLayer = new double[numInputs];
            this.outputLayer = new double[numOutputs];
            this.hiddenLayer = new double[numHidden];
        }

        public double[] Compute(double[] input)
        {
            double[] hiddenValues = new double[numHidden];
            for (int j = 0; j < numHidden; j++)
            {
                double sum = 0;
                for (int k = 0; k < numInputs; k++)
                {
                    sum += (input[k] * ihWeights[k][j]);
                }
                hiddenValues[j] = Math.Tanh(sum + bias[0]);
            }
           

            double[] outputValues = new double[numOutputs];
            for (int j = 0; j < numOutputs; j++)
            {
                double sum = 0;
                for (int k = 0; k < numHidden; k++)
                {
                    sum += (hiddenValues[k] * hoWeights[k][j]);
                }
                outputValues[j] = Math.Tanh(sum + bias[1]);
            }
           

            return outputValues;
        }

        // update the weights
        public void UpdateWeights(double[] data)
        {
            int pos = 0;
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numHidden; j++)
                {
                    ihWeights[i][j] = data[pos];
                    pos++;
                }
            }

            for (int i = 0; i < numHidden; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    hoWeights[i][j] = data[pos];
                    pos++;
                }
            }

            bias[0] = data[pos];
            pos++;
            bias[1] = data[pos];
        }

        // get individual for the population
        public double[] GetWeights()
        {
            double[] weights = new double[(numInputs * numHidden) + (numHidden * numOutputs) + 2];

            int pos = 0;

            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numHidden; j++)
                {
                    weights[pos] = ihWeights[i][j];
                    pos++;
                }
            }

            for (int i = 0; i < numHidden; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    weights[pos] = hoWeights[i][j];
                    pos++;
                }
            }

            weights[weights.Length - 2] = bias[0];
            weights[weights.Length - 1] = bias[1];

            return weights;
        }

        // evaluate
        public double[] Evaluate(double[][] inputX, double[][] outputY)
        {
            double[] rmseList = new double[inputX.Length];

            for (int i = 0; i < inputX.Length; i++)
            {
                double[] PredictedOutput = Compute(inputX[i]);
                double[] ExpectedOutput = outputY[i];

                rmseList[i] = RMSE(PredictedOutput, ExpectedOutput);
            }

            return rmseList;
        }

        // RMSE
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
    }
}
