using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StockPricePrediction
{
    class ACO_NN
    {
        private int numWeights;
        private int numBias;
        private int d;
        private double[][][] discretePoint;
        private double[][][] pheromoneTrails;


        public ACO_NN(int numWeights, int numBias, int d)
        {
            this.d = d;
            this.numBias = numBias;
            this.numWeights = numWeights;

            discretePoint = new double[numWeights + numBias][][];
            pheromoneTrails = new double[numWeights + numWeights][][];

            for (int i = 0; i < discretePoint.Length; i++)
            {
                discretePoint[i] = new double[discretePoint.Length][];
                for (int j = 0; j < discretePoint.Length; j++)
                {
                    discretePoint[i][j] = new double[d];
                    for (int k = 0; k < d; k++)
                    {
                        discretePoint[i][j][k] = 1 / numWeights; ;
                    }

                }

                pheromoneTrails[i] = new double[discretePoint.Length][];
                for (int j = 0; j < pheromoneTrails.Length; j++)
                {
                    pheromoneTrails[i][j] = new double[d];
                    for (int k = 0; k < d; k++)
                    {
                        pheromoneTrails[i][j][k] = 1 / numWeights; ;
                    }
                }
            }
        }

        public void ProbabilisticSolutionConstruction()
        {

        }

    }
}
