using System;
using System.Linq;

namespace SimpleNeuralNetwork
{
    public class SimpleNN
    {
        static Random rnd = new Random();

        // Network params
        const int nIn = 2;
        const int nHidden = 9;
        const int nOut = 1;

        const double lr = 0.01;
        const int epochs = 3696;
        const int samples = 3696;

        // Weighs
        static double[,] W1 = new double[nIn, nHidden];
        static double[] b1 = new double[nHidden];
        static double[,] W2 = new double[nHidden, nOut];
        static double[] b2 = new double[nOut];

        public static void Main()
        {
            // Weighs initialization
            InitWeights(W1); 
            InitWeights(W2);

            // Training data
            Console.WriteLine($"Preparing training data - {samples} samples...");
            double[,] X = new double[samples, nIn];
            double[,] Y = new double[samples, nOut];
            for (int i = 0; i < samples; i++)
            {
                X[i, 0] = rnd.NextDouble();
                X[i, 1] = rnd.NextDouble(); 
                Y[i, 0] = X[i, 0] + X[i, 1];

                Console.WriteLine($"{X[i, 0]:F5} + {X[i, 1]:F5} = {Y[i, 0]:F5}");
            }

            Console.WriteLine($"{Y.Length} training samples ready!" + Environment.NewLine);

            // Training
            Console.WriteLine($"Starting {nHidden} neurons training ({epochs} epochs)...");
            for (int e = 0; e < epochs; e++)
            {
                double loss = 0.0;
                for (int i = 0; i < samples; i++)
                {
                    // forward
                    double[] x = { X[i, 0], X[i, 1] };
                    double[] z1 = Add(MatVec(W1, x), b1);
                    double[] a1 = ReLU(z1);
                    double[] z2 = Add(MatVec(W2, a1), b2);

                    // loss
                    double[] y = { Y[i, 0] };
                    loss += Math.Pow(z2[0] - y[0], 2);

                    // backward
                    double[] gradOut = { 2 * (z2[0] - y[0]) };
                    double[,] dW2 = Outer(a1, gradOut);
                    double[] db2 = gradOut;

                    double[] gradA1 = MatVecT(W2, gradOut);
                    double[] gradZ1 = new double[nHidden];
                    for (int k = 0; k < nHidden; k++)
                        gradZ1[k] = gradA1[k] * (z1[k] > 0 ? 1 : 0);

                    double[,] dW1 = Outer(x, gradZ1);
                    double[] db1 = gradZ1;

                    // update
                    Update(W2, dW2, lr); AddInPlace(b2, db2, -lr);
                    Update(W1, dW1, lr); AddInPlace(b1, db1, -lr);
                }

                if (e % 1000 == 0)
                    Console.WriteLine($"Epoch {e}, Loss {(loss / samples):F5}");
            }
            Console.WriteLine("Training complete!" + Environment.NewLine);

            for (var i = 0; i < W1.GetLength(0); i++)
            {
                var st = string.Empty;
                for (var j = 0; j < W1.GetLength(1); j++)
                {
                    st = string.IsNullOrEmpty(st)
                        ? $"{W1[i, j]:F5}"
                        : $"{st}, {W1[i, j]:F5}";
                }
                Console.WriteLine(st);
            }

            Console.WriteLine();            

            for (var i = 0; i < b1.Length; i++)
            {
                Console.WriteLine($"{b1[i]:F5}");
            }

            Console.WriteLine();

            for (var i = 0; i < W2.GetLength(0); i++)
            {
                var st = string.Empty;
                for (var j = 0; j < W2.GetLength(1); j++)
                {
                    st = string.IsNullOrEmpty(st)
                        ? $"{W2[i, j]:F5}"
                        : $"{st}, {W2[i, j]:F5}";
                }
                Console.WriteLine(st);
            }

            Console.WriteLine();

            for (var i = 0; i < b2.Length; i++)
            {
                Console.WriteLine($"{b2[i]:F5}");
            }

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine("Testing:" + Environment.NewLine);

            // Testing
            double[] test = { 7.0, 3.5 };
            double[] a1t = ReLU(Add(MatVec(W1, test), b1));
            Console.WriteLine($"{string.Join(',', a1t)}{Environment.NewLine}");

            double[] outp = Add(MatVec(W2, a1t), b2);
            double correctResult = test[0] + test[1];

            var resultLine = outp[0] == correctResult
                ? $"{test[0]} + {test[1]} = {outp[0]:F5} CORRECT!"
                : $"{test[0]} + {test[1]} = {outp[0]:F5} MISTAKEN! | correct result: {(test[0] + test[1]):F5}";

            Console.WriteLine(resultLine);
        }

        // --- Helper methods ---
        static void InitWeights(double[,] W)
        {
            for (int i = 0; i < W.GetLength(0); i++)
                for (int j = 0; j < W.GetLength(1); j++)
                    W[i, j] = rnd.NextDouble() * 0.1;
        }

        static double[] MatVec(double[,] W, double[] x)
        {
            int m = W.GetLength(0), n = W.GetLength(1);
            double[] y = new double[n];
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++)
                    y[j] += W[i, j] * x[i];
            return y;
        }

        static double[] MatVecT(double[,] W, double[] x)
        {
            int m = W.GetLength(0), n = W.GetLength(1);
            double[] y = new double[m];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    y[i] += W[i, j] * x[j];
            return y;
        }

        static double[] Add(double[] a, double[] b)
        {
            double[] r = new double[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = a[i] + b[i];
            return r;
        }

        static void AddInPlace(double[] a, double[] b, double scale)
        {
            for (int i = 0; i < a.Length; i++) a[i] += scale * b[i];
        }

        static double[] ReLU(double[] x)
        {
            double[] y = new double[x.Length];
            for (int i = 0; i < x.Length; i++) y[i] = Math.Max(0, x[i]);
            return y;
        }

        static double[,] Outer(double[] a, double[] b)
        {
            double[,] r = new double[a.Length, b.Length];
            for (int i = 0; i < a.Length; i++)
                for (int j = 0; j < b.Length; j++)
                    r[i, j] = a[i] * b[j];
            return r;
        }

        static void Update(double[,] W, double[,] grad, double lr)
        {
            for (int i = 0; i < W.GetLength(0); i++)
                for (int j = 0; j < W.GetLength(1); j++)
                    W[i, j] -= lr * grad[i, j];
        }
    }

}
