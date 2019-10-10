/**
 * If you wish to use this implementation in anyway, please use the following citations
 * 
 * 1. Pillay, Bradley J., and Absalom E. Ezugwu. "Stock Price Forecasting Using Symbiotic Organisms Search Trained Neural Networks." In International Conference on Computational Science and Its Applications, pp. 673-688. Springer, Cham, 2019.
 * 2. Pillay, Bradley J., and Absalom E. Ezugwu. "On the performance of metaheuristics-trained neural networks for improved stock price prediction." article submitted to Neurocomputing - Journal, Elsevier.
 */
using Extreme.DataAnalysis;
using Extreme.Mathematics;
using Extreme.Statistics.TimeSeriesAnalysis;
using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace StockPricePrediction
{
    public partial class Output : Form
    {
        List<double[]> TrainingData;
        List<double[]> TestingData;

        private int[] operators;
        private double[] mape;
        private double[] rmse;
        private double[] mad;
        private double[] mse;
        private double[] mae;

        private string path;
        private string filename;
        Algorithm algorithm;
        private Series openActualForecast = new Series("Actual Forecast (Open Values)");
        private Series openPredictedForecast = new Series("Predicted Forecast (Open Values)");
        private Series closeActualForecast = new Series("Actual Forecast (Close Values)");
        private Series closePredictedForecast = new Series("Predicted Forecast (Close Values)");

        public Output(string path, string filename, Algorithm algorithm)
        {
            InitializeComponent();
            label5.Text = filename;

            this.path = path;
            this.algorithm = algorithm;
            this.filename = filename;
            this.operators = new int[] { 2, 1, 1 };

            LoadData();


            RunAlgorithm();

            

        }

        public Output(string path, string filename, Algorithm algorithm, int[] operators)
        {
            InitializeComponent();
            label5.Text = filename;

            this.path = path;
            this.algorithm = algorithm;
            this.filename = filename;
            this.operators = operators;

            LoadData();
            RunAlgorithm();
        }

        public void LoadData()
        {
            DataSet dataSet = new DataSet(path);

            TrainingData = dataSet.GetTrainingData();
            TestingData = dataSet.GetTestingData();
        }

        public void RunAlgorithm()
        {
            

            if (algorithm == Algorithm.GAFFNN)
            {
                GA();
            }
            else if (algorithm == Algorithm.SOSFFNN)
            {
                SOS();
            }
            else if (algorithm == Algorithm.PSOFFNN)
            {
                PSO();
            }
            else if (algorithm == Algorithm.ARIMA)
            {
                RunARIMA();
            }
        }

        public void GA()
        {
            GA_FFNN ga = new GA_FFNN(30, TrainingData.Take(TrainingData.Count() - 1).ToArray(), TrainingData.Skip(1).ToArray());
            for (int i = 0; i < 1000; i++)
            {
                ga.EvolvePopulation();
            }

            var ind = ga.GetBestIndividual();

            NeuralNetwork nn = new NeuralNetwork(2, 2, 8);
            nn.UpdateWeights(ind);
            nn.Evaluate(TestingData.Take(TestingData.Count() - 1).ToArray(), TestingData.Skip(1).ToArray());
            List<string> display = new List<string>();

            label6.Text = "GA + FEEDFOWARD NEURAL NETWORK";

            List<int> OpenActualLabelsX = new List<int>();
            List<int> OpenPredictedLabelsX = new List<int>();
            List<double> OpenActualData = new List<double>();
            List<double> OpenPredictedData = new List<double>();

            List<int> CloseActualLabelsX = new List<int>();
            List<int> ClosePredictedLabelsX = new List<int>();
            List<double> CloseActualData = new List<double>();
            List<double> ClosePredictedData = new List<double>();

            //
            int c1 = 0;
            foreach (var item in TrainingData.Select(x => x[0]).ToArray())
            {
                c1++;
                OpenActualLabelsX.Add(c1);
                OpenActualData.Add(item);
            }

            int d1 = c1;
            foreach (var item in TestingData.Select(x => x[0]).ToArray())
            {
                d1++;
                OpenActualLabelsX.Add(d1);
                OpenActualData.Add(item);
            }

            d1 = c1;

            //
            int c2 = 0;
            foreach (var item in TrainingData.Select(x => x[1]).ToArray())
            {
                c2++;
                CloseActualLabelsX.Add(c2);
                CloseActualData.Add(item);
            }

            int d2 = c2;
            foreach (var item in TestingData.Select(x => x[1]).ToArray())
            {
                d2++;
                CloseActualLabelsX.Add(d2);
                CloseActualData.Add(item);
            }

            rmse = new double[] { 0, 0 };
            mad = new double[] { 0, 0 };
            mape = new double[] { 0, 0 };
            mae = new double[] { 0, 0 };
            mse = new double[] { 0, 0 };

            List<double> open = new List<double>();
            List<double> close = new List<double>();
            
            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                d1++;
                double[] prediction = nn.Compute(TestingData[i]);
                //display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Predicted Output: (" + predicted[0][i].ToString("F3") + ", " + predicted[1][i].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
                OpenPredictedLabelsX.Add(d1);
                OpenPredictedData.Add(prediction[0]);

                ClosePredictedLabelsX.Add(d1);
                ClosePredictedData.Add(prediction[1]);

                rmse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                rmse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                mse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mae[0] += Math.Abs((TestingData[i + 1][0] - prediction[0]));
                mae[1] += Math.Abs((TestingData[i + 1][1] - prediction[1]));

                mape[0] += (Math.Abs((TestingData[i + 1][0] - prediction[0])) / Math.Abs(TestingData[i + 1][0]));
                mape[1] += (Math.Abs((TestingData[i + 1][1] - prediction[1])) / Math.Abs(TestingData[i + 1][1]));

                open.Add(prediction[0]);
                close.Add(prediction[1]);
            }

            double openMean = open.Average();
            double closeMean = close.Average();

            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                mad[0] += Math.Abs((open[0] - openMean));
                mad[1] += Math.Abs((close[1] - closeMean));
            }

            mse[0] /= TestingData.Count() - 1;
            mse[1] /= TestingData.Count() - 1;

            mae[0] = (mae[0] / (TestingData.Count() - 1));
            mae[1] = (mae[1] / (TestingData.Count() - 1));


            rmse[0] /= TestingData.Count() - 1;
            rmse[1] /= TestingData.Count() - 1;
            rmse[0] = Math.Sqrt(rmse[0]);
            rmse[1] = Math.Sqrt(rmse[1]);

            mape[0] = (mape[0] / (TestingData.Count() - 1)) * 100;
            mape[1] = (mape[1] / (TestingData.Count() - 1)) * 100;

            mad[0] = (mad[0] / (TestingData.Count() - 1));
            mad[1] = (mad[1] / (TestingData.Count() - 1));

           
            openActualForecast.Points.DataBindXY(OpenActualLabelsX.ToArray(), OpenActualData.ToArray());
            openActualForecast.ChartType = SeriesChartType.Line;
            openPredictedForecast.Points.DataBindXY(OpenPredictedLabelsX.ToArray(), OpenPredictedData.ToArray());
            openPredictedForecast.ChartType = SeriesChartType.Line;
            openActualForecast.Color = Color.RoyalBlue;
            openPredictedForecast.Color = Color.Red;


            closeActualForecast.Points.DataBindXY(CloseActualLabelsX.ToArray(), CloseActualData.ToArray());
            closeActualForecast.ChartType = SeriesChartType.Line;
            closePredictedForecast.Points.DataBindXY(ClosePredictedLabelsX.ToArray(), ClosePredictedData.ToArray());
            closePredictedForecast.ChartType = SeriesChartType.Line;
            closeActualForecast.Color = Color.RoyalBlue;
            closePredictedForecast.Color = Color.Red;

            chart2.Series.Clear();
            chart2.Series.Add(openActualForecast);
            chart2.Series.Add(openPredictedForecast);
            chart2.ChartAreas[0].AxisX.Title = "Day";
            chart2.ChartAreas[0].AxisY.Title = "Normalized Stock Price";

            //label7.Text = (Selection)(operators[0]) + " Selection + " + (Crossover)(operators[1]) + " Crossover + " + (Mutation)(operators[2]) + " Mutations";
            //double[][] predictedValues = new double[rmse.Length][];
            //for (int i = 0; i < rmse.Length; i++)
            //{
            //    double[] prediction = nn.Compute(TestingData[i]);
            //    predictedValues[i] = prediction;
            //    display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i + 1][0].ToString("F3") + ", " + TestingData[i + 1][1].ToString("F3") + ")   |   Predicted Output: (" + prediction[0].ToString("F3") + ", " + prediction[1].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
            //}
            //textBox1.Lines = display.ToArray();
            //double[] mape = MAPE(TestingData.Skip(1).ToArray(), predictedValues);
            //double mad = MAD(TestingData.Skip(1).ToArray(), predictedValues);

            textBox4.Text = mad[0].ToString().Replace(",", ".");
            textBox3.Text = mape[0].ToString().Replace(",", ".");
            textBox2.Text = rmse[0].ToString().Replace(",", ".");
            textBox1.Text = mse[0].ToString().Replace(",", ".");
            textBox5.Text = mae[0].ToString().Replace(",", ".");
        }

        public void SOS()
        {
            SOS_NN sos_NN = new SOS_NN(100, 30, new NeuralNetwork(2, 2, 8), TrainingData.Take(TrainingData.Count() - 1).ToArray(), TrainingData.Skip(1).ToArray());

            sos_NN.Train();
            sos_NN.UpdateNeuralNetworkWithBestWeights();
            

            NeuralNetwork nn = sos_NN.GetNeuralNetwork();
            rmse = nn.Evaluate(TestingData.Take(TestingData.Count() - 1).ToArray(), TestingData.Skip(1).ToArray());
            List<string> display = new List<string>();
            label6.Text = "SOS + FEEDFOWARD NEURAL NETWORK";

            List<int> OpenActualLabelsX = new List<int>();
            List<int> OpenPredictedLabelsX = new List<int>();
            List<double> OpenActualData = new List<double>();
            List<double> OpenPredictedData = new List<double>();

            List<int> CloseActualLabelsX = new List<int>();
            List<int> ClosePredictedLabelsX = new List<int>();
            List<double> CloseActualData = new List<double>();
            List<double> ClosePredictedData = new List<double>();

            //
            int c1 = 0;
            foreach (var item in TrainingData.Select(x => x[0]).ToArray())
            {
                c1++;
                OpenActualLabelsX.Add(c1);
                OpenActualData.Add(item);
            }

            int d1 = c1;
            foreach (var item in TestingData.Select(x => x[0]).ToArray())
            {
                d1++;
                OpenActualLabelsX.Add(d1);
                OpenActualData.Add(item);
            }

            d1 = c1;

            //
            int c2 = 0;
            foreach (var item in TrainingData.Select(x => x[1]).ToArray())
            {
                c2++;
                CloseActualLabelsX.Add(c2);
                CloseActualData.Add(item);
            }

            int d2 = c2;
            foreach (var item in TestingData.Select(x => x[1]).ToArray())
            {
                d2++;
                CloseActualLabelsX.Add(d2);
                CloseActualData.Add(item);
            }

            rmse = new double[] { 0, 0 };
            mad = new double[] { 0, 0 };
            mape = new double[] { 0, 0 };
            mae = new double[] { 0, 0 };
            mse = new double[] { 0, 0 };

            List<double> open = new List<double>();
            List<double> close = new List<double>();

            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                d1++;
                double[] prediction = nn.Compute(TestingData[i]);
                //display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Predicted Output: (" + predicted[0][i].ToString("F3") + ", " + predicted[1][i].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
                OpenPredictedLabelsX.Add(d1);
                OpenPredictedData.Add(prediction[0]);

                ClosePredictedLabelsX.Add(d1);
                ClosePredictedData.Add(prediction[1]);

                rmse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                rmse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                mse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mae[0] += Math.Abs((TestingData[i + 1][0] - prediction[0]));
                mae[1] += Math.Abs((TestingData[i + 1][1] - prediction[1]));

                mape[0] += (Math.Abs((TestingData[i + 1][0] - prediction[0])) / Math.Abs(TestingData[i + 1][0]));
                mape[1] += (Math.Abs((TestingData[i + 1][1] - prediction[1])) / Math.Abs(TestingData[i + 1][1]));

                open.Add(prediction[0]);
                close.Add(prediction[1]);
            }

            double openMean = open.Average();
            double closeMean = close.Average();

            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                mad[0] += Math.Abs((open[0] - openMean));
                mad[1] += Math.Abs((close[1] - closeMean));
            }
            mse[0] /= TestingData.Count() - 1;
            mse[1] /= TestingData.Count() - 1;

            mae[0] = (mae[0] / (TestingData.Count() - 1));
            mae[1] = (mae[1] / (TestingData.Count() - 1));

            rmse[0] /= (TestingData.Count()  - 1);
            rmse[1] /=  (TestingData.Count() - 1);
            rmse[0] = Math.Sqrt(rmse[0]);
            rmse[1] = Math.Sqrt(rmse[1]);

            mape[0] = (mape[0] / (TestingData.Count() - 1)) * 100;
            mape[1] = (mape[1] / (TestingData.Count() - 1)) * 100;

            mad[0] = (mad[0] / (TestingData.Count() - 1));
            mad[1] = (mad[1] / (TestingData.Count() - 1));

            openActualForecast.Points.DataBindXY(OpenActualLabelsX.ToArray(), OpenActualData.ToArray());
            openActualForecast.ChartType = SeriesChartType.Line;
            openPredictedForecast.Points.DataBindXY(OpenPredictedLabelsX.ToArray(), OpenPredictedData.ToArray());
            openPredictedForecast.ChartType = SeriesChartType.Line;
            openActualForecast.Color = Color.RoyalBlue;
            openPredictedForecast.Color = Color.Red;


            closeActualForecast.Points.DataBindXY(CloseActualLabelsX.ToArray(), CloseActualData.ToArray());
            closeActualForecast.ChartType = SeriesChartType.Line;
            closePredictedForecast.Points.DataBindXY(ClosePredictedLabelsX.ToArray(), ClosePredictedData.ToArray());
            closePredictedForecast.ChartType = SeriesChartType.Line;
            closeActualForecast.Color = Color.RoyalBlue;
            closePredictedForecast.Color = Color.Red;

            chart2.Series.Clear();
            chart2.Series.Add(openActualForecast);
            chart2.Series.Add(openPredictedForecast);
            chart2.ChartAreas[0].AxisX.Title = "Day";
            chart2.ChartAreas[0].AxisY.Title = "Normalized Stock Price";

            //double[][] predictedValues = new double[rmse.Length][];
            //for (int i = 0; i < rmse.Length; i++)
            //{
            //    double[] prediction = nn.Compute(TestingData[i]);
            //    predictedValues[i] = prediction;
            //    display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i+1][0].ToString("F3") + ", " + TestingData[i+1][1].ToString("F3") + ")   |   Predicted Output: (" + prediction[0].ToString("F3") + ", " + prediction[1].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
            //}
            //textBox1.Lines = display.ToArray();

            //double[] mape = MAPE(TestingData.Skip(1).ToArray(), predictedValues);
            //double mad = MAD(TestingData.Skip(1).ToArray(), predictedValues);

            textBox4.Text = mad[0].ToString().Replace(",", ".");
            textBox3.Text = mape[0].ToString().Replace(",", "."); ;
            textBox2.Text = rmse[0].ToString().Replace(",", ".");
            textBox1.Text = mse[0].ToString().Replace(",", ".");
            textBox5.Text = mae[0].ToString().Replace(",", ".");
        }

        public void PSO()
        {
            PSO_NN sos_NN = new PSO_NN((2 * 8) + (8 * 2) + 2, 30, 1000, new NeuralNetwork(2, 2, 8), TrainingData.Take(TrainingData.Count() - 1).ToArray(), TrainingData.Skip(1).ToArray());


            sos_NN.Train();
            sos_NN.UpdateNeuralNetworkWithBestWeights();

            NeuralNetwork nn = sos_NN.GetNeuralNetwork();
            nn.Evaluate(TestingData.Take(TestingData.Count() - 1).ToArray(), TestingData.Skip(1).ToArray());
            List<string> display = new List<string>();
            label6.Text = "PSO + FEEDFOWARD NEURAL NETWORK";

            List<int> OpenActualLabelsX = new List<int>();
            List<int> OpenPredictedLabelsX = new List<int>();
            List<double> OpenActualData = new List<double>();
            List<double> OpenPredictedData = new List<double>();

            List<int> CloseActualLabelsX = new List<int>();
            List<int> ClosePredictedLabelsX = new List<int>();
            List<double> CloseActualData = new List<double>();
            List<double> ClosePredictedData = new List<double>();

            //
            int c1 = 0;
            foreach (var item in TrainingData.Select(x => x[0]).ToArray())
            {
                c1++;
                OpenActualLabelsX.Add(c1);
                OpenActualData.Add(item);
            }

            int d1 = c1;
            foreach (var item in TestingData.Select(x => x[0]).ToArray())
            {
                d1++;
                OpenActualLabelsX.Add(d1);
                OpenActualData.Add(item);
            }

            d1 = c1;

            //
            int c2 = 0;
            foreach (var item in TrainingData.Select(x => x[1]).ToArray())
            {
                c2++;
                CloseActualLabelsX.Add(c2);
                CloseActualData.Add(item);
            }

            int d2 = c2;
            foreach (var item in TestingData.Select(x => x[1]).ToArray())
            {
                d2++;
                CloseActualLabelsX.Add(d2);
                CloseActualData.Add(item);
            }

            rmse = new double[] { 0, 0 };
            mad = new double[] { 0, 0 };
            mape = new double[] { 0, 0 };
            mae = new double[] { 0, 0 };
            mse = new double[] { 0, 0 };

            List<double> open = new List<double>();
            List<double> close = new List<double>();

            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                d1++;
                double[] prediction = nn.Compute(TestingData[i]);
                //display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Predicted Output: (" + predicted[0][i].ToString("F3") + ", " + predicted[1][i].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
                OpenPredictedLabelsX.Add(d1);
                OpenPredictedData.Add(prediction[0]);

                ClosePredictedLabelsX.Add(d1);
                ClosePredictedData.Add(prediction[1]);

                rmse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                rmse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mse[0] += Math.Pow(TestingData[i + 1][0] - prediction[0], 2);
                mse[1] += Math.Pow(TestingData[i + 1][1] - prediction[1], 2);

                mae[0] += Math.Abs((TestingData[i + 1][0] - prediction[0]));
                mae[1] += Math.Abs((TestingData[i + 1][1] - prediction[1]));

                mape[0] += (Math.Abs((TestingData[i + 1][0] - prediction[0])) / Math.Abs(TestingData[i + 1][0]));
                mape[1] += (Math.Abs((TestingData[i + 1][1] - prediction[1])) / Math.Abs(TestingData[i + 1][1]));

                open.Add(prediction[0]);
                close.Add(prediction[1]);
            }

            double openMean = open.Average();
            double closeMean = close.Average();

            for (int i = 0; i < TestingData.Count() - 1; i++)
            {
                mad[0] += Math.Abs((open[0] - openMean));
                mad[1] += Math.Abs((close[1] - closeMean));
            }

            mse[0] /= TestingData.Count() - 1;
            mse[1] /= TestingData.Count() - 1;

            mae[0] = (mae[0] / (TestingData.Count() - 1));
            mae[1] = (mae[1] / (TestingData.Count() - 1));

            rmse[0] /= TestingData.Count() - 1;
            rmse[1] /= TestingData.Count() - 1;
            rmse[0] = Math.Sqrt(rmse[0]);
            rmse[1] = Math.Sqrt(rmse[1]);

            mape[0] = (mape[0] / (TestingData.Count() - 1)) * 100;
            mape[1] = (mape[1] / (TestingData.Count() - 1)) * 100;

            mad[0] = (mad[0] / (TestingData.Count() - 1));
            mad[1] = (mad[1] / (TestingData.Count() - 1));

            openActualForecast.Points.DataBindXY(OpenActualLabelsX.ToArray(), OpenActualData.ToArray());
            openActualForecast.ChartType = SeriesChartType.Line;
            openPredictedForecast.Points.DataBindXY(OpenPredictedLabelsX.ToArray(), OpenPredictedData.ToArray());
            openPredictedForecast.ChartType = SeriesChartType.Line;
            openActualForecast.Color = Color.RoyalBlue;
            openPredictedForecast.Color = Color.Red;


            closeActualForecast.Points.DataBindXY(CloseActualLabelsX.ToArray(), CloseActualData.ToArray());
            closeActualForecast.ChartType = SeriesChartType.Line;
            closePredictedForecast.Points.DataBindXY(ClosePredictedLabelsX.ToArray(), ClosePredictedData.ToArray());
            closePredictedForecast.ChartType = SeriesChartType.Line;
            closeActualForecast.Color = Color.RoyalBlue;
            closePredictedForecast.Color = Color.Red;

            chart2.Series.Clear();
            chart2.Series.Add(openActualForecast);
            chart2.Series.Add(openPredictedForecast);
            chart2.ChartAreas[0].AxisX.Title = "Day";
            chart2.ChartAreas[0].AxisY.Title = "Normalized Stock Price";

            textBox4.Text = mad[0].ToString().Replace(",", ".");
            textBox3.Text = mape[0].ToString().Replace(",", ".");
            textBox2.Text = rmse[0].ToString().Replace(",", ".");
            textBox1.Text = mse[0].ToString().Replace(",", ".");
            textBox5.Text = mae[1].ToString().Replace(",", ".");
        }

        public void RunARIMA()
        {
            ARIMA aRIMA = new ARIMA(TrainingData, operators);
            double[][] predicted = aRIMA.Forecast(TestingData.Count());
            rmse = aRIMA.Evaluate_RMSE(predicted, TestingData.ToArray());
            List<string> display = new List<string>();
            label6.Text = "ARIMA";
            
            List<int> OpenActualLabelsX = new List<int>();
            List<int> OpenPredictedLabelsX = new List<int>();
            List<double> OpenActualData = new List<double>();
            List<double> OpenPredictedData = new List<double>();

            List<int> CloseActualLabelsX = new List<int>();
            List<int> ClosePredictedLabelsX = new List<int>();
            List<double> CloseActualData = new List<double>();
            List<double> ClosePredictedData = new List<double>();

            //
            int c1 = 0;
            foreach (var item in TrainingData.Select(x => x[0]).ToArray())
            {
                c1++;
                OpenActualLabelsX.Add(c1);
                OpenActualData.Add(item);
            }

            int d1 = c1;
            foreach (var item in TestingData.Select(x => x[0]).ToArray())
            {
                d1++;
                OpenActualLabelsX.Add(d1);
                OpenActualData.Add(item);
            }

            d1 = c1;  

            for (int i = 0; i < predicted[0].Length; i++)
            {
                d1++;
                //display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Predicted Output: (" + predicted[0][i].ToString("F3") + ", " + predicted[1][i].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
                OpenPredictedLabelsX.Add(d1);
                OpenPredictedData.Add(predicted[0][i]);
            }
            
            openActualForecast.Points.DataBindXY(OpenActualLabelsX.ToArray(), OpenActualData.ToArray());
            openActualForecast.ChartType = SeriesChartType.Line;
            openPredictedForecast.Points.DataBindXY(OpenPredictedLabelsX.ToArray(), OpenPredictedData.ToArray());
            openPredictedForecast.ChartType = SeriesChartType.Line;
            openActualForecast.Color = Color.RoyalBlue;
            openPredictedForecast.Color = Color.Red;

            //
            int c2 = 0;
            foreach (var item in TrainingData.Select(x => x[1]).ToArray())
            {
                c2++;
                CloseActualLabelsX.Add(c2);
                CloseActualData.Add(item);
            }

            int d2 = c2;
            foreach (var item in TestingData.Select(x => x[1]).ToArray())
            {
                d2++;
                CloseActualLabelsX.Add(d2);
                CloseActualData.Add(item);
            }

            d2 = c2;

            for (int i = 0; i < predicted[0].Length; i++)
            {
                d2++;
                //display.Add("Input: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Expected Output: (" + TestingData[i][0].ToString("F3") + ", " + TestingData[i][1].ToString("F3") + ")   |   Predicted Output: (" + predicted[0][i].ToString("F3") + ", " + predicted[1][i].ToString("F3") + ")  |  RMSE: " + rmse[i].ToString("F3"));
                ClosePredictedLabelsX.Add(d2);
                ClosePredictedData.Add(predicted[1][i]);
            }

            closeActualForecast.Points.DataBindXY(CloseActualLabelsX.ToArray(), CloseActualData.ToArray());
            closeActualForecast.ChartType = SeriesChartType.Line;
            closePredictedForecast.Points.DataBindXY(ClosePredictedLabelsX.ToArray(), ClosePredictedData.ToArray());
            closePredictedForecast.ChartType = SeriesChartType.Line;
            closeActualForecast.Color = Color.RoyalBlue;
            closePredictedForecast.Color = Color.Red;


            chart2.Series.Clear();
            chart2.Series.Add(openActualForecast);
            chart2.Series.Add(openPredictedForecast);
            chart2.ChartAreas[0].AxisX.Title = "Day";
            chart2.ChartAreas[0].AxisY.Title = "Normalized Stock Price";

            //chart2.ChartAreas[0].AxisX.LabelStyle.ForeColor = Color.White;


            //textBox1.Lines = display.ToArray();
            mape = aRIMA.Evaluate_MAPE(predicted, TestingData.ToArray());
            mad = aRIMA.Evaluate_MAD(predicted, TestingData.ToArray());
            mse = aRIMA.Evaluate_MSE(predicted, TestingData.ToArray());
            mae = aRIMA.Evaluate_MAE(predicted, TestingData.ToArray());

            textBox4.Text = mad[0].ToString().Replace(",", ".");
            textBox3.Text = mape[0].ToString().Replace(",", "."); 
            textBox2.Text = rmse[0].ToString().Replace(",", ".");
            textBox1.Text = mse[0].ToString().Replace(",", ".");
            textBox5.Text = mae[0].ToString().Replace(",", ".");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            new ChooseAlgorithm(path, filename).Show();
            this.Hide();
        }

        private void label5_Click(object sender, EventArgs e)
        {

        }

        private void button7_Click(object sender, EventArgs e)
        {
            new Output(path, filename, algorithm, operators).Show();
            this.Hide();
        }

        public double[] MAPE(double[][] Actual, double[][] Predicted)
        {
            double[] mape = new double[] { 0, 0, 0 };
            for (int i = 0; i < Actual.Length; i++)
            {
                mape[0] += Math.Abs((Actual[i][0] - Predicted[i][0])) / Math.Abs(Actual[i][0]);
                mape[1] += Math.Abs((Actual[i][1] - Predicted[i][1])) / Math.Abs(Actual[i][1]);
            }

            mape[0] = (mape[0] / Actual.Length) * 100;
            mape[1] = (mape[1] / Actual.Length) * 100;

            mape[2] = (mape[0] + mape[1]) / 2;

            return mape;
        }

        public double MAD(double[][] Actual, double[][] Predicted)
        {
            double[] MAD = new double[] { 0, 0, 0 };

            List<double> open = new List<double>();
            List<double> close = new List<double>();
            for (int i = 0; i < Actual.Length; i++)
            {
                open.Add(Predicted[i][0]);
                close.Add(Predicted[i][1]);
            }

            double openMean = open.Average();
            double closeMean = close.Average();

            for (int i = 0; i < Actual.Length; i++)
            {
                MAD[0] += Math.Abs((Predicted[i][0] - openMean));
                MAD[1] += Math.Abs((Predicted[i][1] - closeMean));
            }

            MAD[0] = (MAD[0] / Actual.Length);
            MAD[1] = (MAD[1] / Actual.Length);

            MAD[2] = (MAD[0] + MAD[1]) / 2;

            Console.WriteLine("MAD");

            return MAD[2];
        }

        private void button2_Click(object sender, EventArgs e)
        {
            chart2.Series.Clear();
            chart2.Series.Add(openActualForecast);
            chart2.Series.Add(openPredictedForecast);
            textBox3.Text = mape[0].ToString().Replace(",", ".");
            textBox2.Text = rmse[0].ToString().Replace(",", ".");
            textBox4.Text = mad[0].ToString().Replace(",", ".");
            textBox1.Text = mse[0].ToString().Replace(",", ".");
            textBox5.Text = mae[0].ToString().Replace(",", ".");
        }

        private void button3_Click(object sender, EventArgs e)
        {
            chart2.Series.Clear();
            chart2.Series.Add(closeActualForecast);
            chart2.Series.Add(closePredictedForecast);
            textBox3.Text = mape[1].ToString().Replace(",", ".");
            textBox2.Text = rmse[1].ToString().Replace(",", ".");
            textBox4.Text = mad[1].ToString().Replace(",", ".");
            textBox1.Text = mse[1].ToString().Replace(",", ".");
            textBox5.Text = mae[1].ToString().Replace(",", ".");
        }

        private void Output_Load(object sender, EventArgs e)
        {

        }
    }

    public enum Algorithm
    {
        ARIMA,
        GAFFNN,
        SOSFFNN,
        PSOFFNN,
        ACOFFNN
    }
}
