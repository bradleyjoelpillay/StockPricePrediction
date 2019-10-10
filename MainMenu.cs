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
    public partial class MainMenu : Form
    {
        private string path;

        public MainMenu()
        {
            InitializeComponent();
        }

        private void MainMenu_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {

            OpenFileDialog file = new OpenFileDialog();
            file.Title = "Choose a dataset";
            file.Multiselect = false;
            file.Filter = "CSV (*.csv)|*.csv";
            if (file.ShowDialog() == DialogResult.OK)
            {
                path = file.FileName;
                if (path.EndsWith(".csv"))
                {
                    int lastIndex = path.LastIndexOf("\\") + 1;
                    int length = path.Length - lastIndex;
                    string fileName = path.Substring(lastIndex, length);
                    label3.Text = fileName;
                }
                
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            
            if (label3.Text == "No File Selected")
            {
                MessageBox.Show("Please choose a file!", "Alert", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else if (textBox1.Text.Trim() == "")
            {
                MessageBox.Show("Enter the name of your dataset!", "Alert", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            else
            {
                new ChooseAlgorithm(path, textBox1.Text).Show();
                this.Hide();
            }
        }
    }
}
