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
    public partial class ChooseAlgorithm : Form
    {
        private string path;
        private string DataSetName;

        public ChooseAlgorithm(string path, string dataset)
        {
            InitializeComponent();
            this.path = path;
            this.DataSetName = dataset;
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void button7_Click(object sender, EventArgs e)
        {
            new MainMenu().Show();
            this.Hide();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.GAFFNN).Show();
            this.Hide();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.SOSFFNN).Show();
            this.Hide();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.PSOFFNN).Show();
            this.Hide();
        }

        private void button1_Click(object sender, EventArgs e)
        {

        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            new ChooseParameters(path, DataSetName).Show();
            this.Hide();
        }

        private void button6_Click(object sender, EventArgs e)
        {

        }

        private void button2_Click_1(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.GAFFNN).Show();
            this.Hide();
        }

        private void button4_Click_1(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.SOSFFNN).Show();
            this.Hide();
        }

        private void button3_Click_1(object sender, EventArgs e)
        {
            new Output(path, DataSetName, Algorithm.PSOFFNN).Show();
            this.Hide();
        }

        private void ChooseAlgorithm_Load(object sender, EventArgs e)
        {

        }
    }
}
