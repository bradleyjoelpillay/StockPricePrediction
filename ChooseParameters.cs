/**
 * If you wish to use this implementation in anyway, please use the following citations
 * 
 * 1. Pillay, Bradley J., and Absalom E. Ezugwu. "Stock Price Forecasting Using Symbiotic Organisms Search Trained Neural Networks." In International Conference on Computational Science and Its Applications, pp. 673-688. Springer, Cham, 2019.
 * 2. Pillay, Bradley J., and Absalom E. Ezugwu. "On the performance of metaheuristics-trained neural networks for improved stock price prediction." article submitted to Neurocomputing - Journal, Elsevier.
 */

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace StockPricePrediction
{
    public partial class ChooseParameters : Form
    {
        private string path { get; set; }
        private string datasetName { get; set; }
        public ChooseParameters(string path, string DatasetName)
        {
            InitializeComponent();
            this.path = path;
            this.datasetName = DatasetName;
        }

        private void ChooseParameters_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            int p = (int)numericUpDown1.Value;
            int d = (int)numericUpDown2.Value;
            int q = (int)numericUpDown3.Value;

            new Output(path, datasetName, Algorithm.ARIMA, new int[] { p, d, q }).Show();
            this.Hide();
        }
    }
}
