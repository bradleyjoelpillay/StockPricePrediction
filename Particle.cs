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
    public class Particle
    {
        public double[] position { get; set; }
        public double error { get; set; }
        public double[] velocity { get; set; }
        public double[] bestPosition { get; set; }
        public double bestError { get; set; }

        public Particle(double[] pos, double err, double[] vel, double[] bestPos, double bestErr)
        {
            this.position = new double[pos.Length];
            pos.CopyTo(this.position, 0);
            this.error = err;
            this.velocity = new double[vel.Length];
            vel.CopyTo(this.velocity, 0);
            this.bestPosition = new double[bestPos.Length];
            bestPos.CopyTo(this.bestPosition, 0);
            this.bestError = bestErr;
        }


    }
}
