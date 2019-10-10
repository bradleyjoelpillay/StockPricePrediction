/**
 * If you wish to use this implementation in anyway, please use the following citations
 * 
 * 1. Pillay, Bradley J., and Absalom E. Ezugwu. "Stock Price Forecasting Using Symbiotic Organisms Search Trained Neural Networks." In International Conference on Computational Science and Its Applications, pp. 673-688. Springer, Cham, 2019.
 * 2. Pillay, Bradley J., and Absalom E. Ezugwu. "On the performance of metaheuristics-trained neural networks for improved stock price prediction." article submitted to Neurocomputing - Journal, Elsevier.
 */
using Microsoft.VisualBasic.FileIO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StockPricePrediction
{
    class DataSet
    {
        private List<double[]> TrainingData;
        private List<double[]> TestingData;

        private double[] MinMaxValuesTrain = new double[] { 0, 0, 0, 0 };
        private double[] MinMaxValuesTest = new double[] { 0, 0, 0, 0 };

        public DataSet(string path)
        {
            TrainingData = new List<double[]>();

            using (TextFieldParser parser = new TextFieldParser(path))
            {
                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(",");

                int i = 0;

                while (!parser.EndOfData)
                {                    
                    if (i > 0)
                    {
                        string[] fields = parser.ReadFields();

                        try
                        {
                            double[] temp = new double[2];

                            temp[0] = double.Parse(fields[1].Trim().Replace(".", ","));
                            temp[1] = double.Parse(fields[4].Trim().Replace(".", ","));
                            TrainingData.Add(temp);
                        }
                        catch (Exception)
                        {

                        }
                    }
                    i++;
                }

            }
            
            // split the data
            int numTrain = (int)Math.Ceiling(TrainingData.Count() * 0.7);
            int numTest = TrainingData.Count() - numTrain;
            TestingData = TrainingData.Skip(numTrain).Take(numTest).ToList();
            TrainingData = TrainingData.Take(numTrain).ToList();

            // find the min and max for open and close
            MinMaxValuesTrain[0] = Math.Floor(TrainingData.Select(x => x[0]).Min()); // Min Open for Train
            MinMaxValuesTrain[1] = Math.Ceiling(TrainingData.Select(x => x[0]).Max()); // Max Open for Train
            MinMaxValuesTrain[2] = Math.Floor(TrainingData.Select(x => x[1]).Min()); // Min Close for Train
            MinMaxValuesTrain[3] = Math.Ceiling(TrainingData.Select(x => x[1]).Max()); // Max Close for Train

            MinMaxValuesTest[0] = Math.Floor(TestingData.Select(x => x[0]).Min()); // Min Open for Test
            MinMaxValuesTest[1] = Math.Ceiling(TestingData.Select(x => x[0]).Max()); // Max Open for Test
            MinMaxValuesTest[2] = Math.Floor(TestingData.Select(x => x[1]).Min()); // Min Close for Test
            MinMaxValuesTest[3] = Math.Ceiling(TestingData.Select(x => x[1]).Max()); // Max Close for Test

            foreach (var item in TrainingData)
            {
                item[0] = (item[0] - MinMaxValuesTrain[0]) / (MinMaxValuesTrain[1] - MinMaxValuesTrain[0]);
                item[1] = (item[1] - MinMaxValuesTrain[2]) / (MinMaxValuesTrain[3] - MinMaxValuesTrain[2]);
            }

            foreach (var item in TestingData)
            {
                item[0] = (item[0] - MinMaxValuesTest[0]) / (MinMaxValuesTest[1] - MinMaxValuesTest[0]);
                item[1] = (item[1] - MinMaxValuesTest[2]) / (MinMaxValuesTest[3] - MinMaxValuesTest[2]);
            }

        }

        public List<double[]> GetTrainingData()
        {
            return TrainingData;
        }

        public List<double[]> GetTestingData()
        {
            return TestingData;
        }

        
    }
}
