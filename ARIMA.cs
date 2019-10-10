/**
 * If you wish to use this implementation in anyway, please use the following citations
 * 
 * 1. Pillay, Bradley J., and Absalom E. Ezugwu. "Stock Price Forecasting Using Symbiotic Organisms Search Trained Neural Networks." In International Conference on Computational Science and Its Applications, pp. 673-688. Springer, Cham, 2019.
 * 2. Pillay, Bradley J., and Absalom E. Ezugwu. "On the performance of metaheuristics-trained neural networks for improved stock price prediction." article submitted to Neurocomputing - Journal, Elsevier.
 */
using Extreme.DataAnalysis;
using Extreme.Mathematics;
using Extreme.Statistics.TimeSeriesAnalysis;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StockPricePrediction
{
    class ARIMA
    {
        private double[] openValues;
        private double[] closeValues;
        private int[] parameters;

        public ARIMA(List<double[]> TrainingData, int[] parameters)
        {
            openValues = TrainingData.Select(x => x[0]).ToArray();
            closeValues = TrainingData.Select(x => x[1]).ToArray();
            this.parameters = parameters;
        }

        public double[][] Forecast(int days)
        {
            var openVector = Vector.Create(openValues);
            var closeVector = Vector.Create(closeValues);

            ArimaModel openModel = new ArimaModel(openVector, parameters[0], parameters[1], parameters[2]);
            ArimaModel closeModel = new ArimaModel(openVector, parameters[0], parameters[1], parameters[2]);

            openModel.Fit();
            closeModel.Fit();

            double[][] forecast = new double[2][];

            forecast[0] = openModel.Forecast(days).ToArray(); // open forecast
            forecast[1] = closeModel.Forecast(days).ToArray(); // close forecast

            return forecast;
        }

        public double[] Evaluate_RMSE(double[][] predicted, // 2 x n array
            double[][] actual // n x 2 array
            )
        {
            double[] rmse = new double[] { 0, 0 };
            for (int i = 0; i < actual.Length; i++)
            {
                
                rmse[0] += Math.Pow((actual[i][0] - predicted[0][i]), 2);
                rmse[1] += Math.Pow((actual[i][1] - predicted[1][i]), 2);
            }

            rmse[0] /= actual.Length;
            rmse[1] /= actual.Length;

            rmse[0] = Math.Sqrt(rmse[0]);
            rmse[1] = Math.Sqrt(rmse[1]);

            return rmse;
        }

        public double[] Evaluate_MSE(double[][] predicted, // 2 x n array
            double[][] actual // n x 2 array
            )
        {
            double[] mse = new double[] { 0, 0 };
            for (int i = 0; i < actual.Length; i++)
            {

                mse[0] += Math.Pow((actual[i][0] - predicted[0][i]), 2);
                mse[1] += Math.Pow((actual[i][1] - predicted[1][i]), 2);
            }

            mse[0] /= actual.Length;
            mse[1] /= actual.Length;

            return mse;
        }

        public double[] Evaluate_MAPE(double[][] predicted, double[][] actual)
        {
            double[] mape = new double[] { 0, 0, 0 };
            for (int i = 0; i < actual.Length; i++)
            {
                mape[0] += (Math.Abs((actual[i][0] - predicted[0][i])) / Math.Abs(actual[i][0]));
                mape[1] += (Math.Abs((actual[i][1] - predicted[1][i])) / Math.Abs(actual[i][1]));
            }
            mape[0] = (mape[0] / actual.Length) * 100;
            mape[1] = (mape[1] / actual.Length) * 100;

            mape[2] = (mape[0] + mape[1]) / 2;
            return mape;
        }

        public double[] Evaluate_MAD(double[][] predicted, double[][] actual)
        {
            double[] MAD = new double[] { 0, 0, 0 };

            List<double> open = new List<double>();
            List<double> close = new List<double>();
            for (int i = 0; i < actual.Length; i++)
            {
                open.Add(predicted[0][i]);
                close.Add(predicted[1][i]);
            }

            double openMean = open.Average();
            double closeMean = close.Average();

            for (int i = 0; i < actual.Length; i++)
            {
                MAD[0] += Math.Abs((predicted[0][i] - openMean));
                MAD[1] += Math.Abs((predicted[1][i] - closeMean));
            }

            MAD[0] = (MAD[0] / actual.Length);
            MAD[1] = (MAD[1] / actual.Length);

            MAD[2] = (MAD[0] + MAD[1]) / 2;

            return MAD;
        }

        public double[] Evaluate_MAE(double[][] predicted, double[][] actual)
        {
            double[] MAD = new double[] { 0, 0, 0 };
            

            for (int i = 0; i < actual.Length; i++)
            {
                MAD[0] += Math.Abs((actual[i][0] - predicted[0][i]));
                MAD[1] += Math.Abs((actual[i][1] - predicted[1][i]));
            }

            MAD[0] = (MAD[0] / actual.Length);
            MAD[1] = (MAD[1] / actual.Length);

            MAD[2] = (MAD[0] + MAD[1]) / 2;

            return MAD;
        }
    }
}
