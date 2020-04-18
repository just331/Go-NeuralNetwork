package Go_NeuralNetwork

import "math"

// y = a(w*x+b)
// dot product of the weight vector (w) and the input vector (x) is added with the bias (b)
// output will be either 0 or 1

func (a *Perceptron) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (a *Perceptron) forwardProb(x []float64) (sum float64) {
	return a.sigmoid(dotProduct(a.weights, x) + a.bias)
}
