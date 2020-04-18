package Go_NeuralNetwork

/*
	Training consists of two parts:
		1.) Backpropagation: algorithm for computing the gradient of the loss function w.r.t the weights
		2.) Loss function: used to get estimation of how far we are from desired solution. Generally mean squared error
			is chosen as loss function
				MSE = 1/n * sum from i=1 to n of (yi - mean(yi))^2
				gradient of loss function calculated using the chain rule

*/

func (a *Perceptron) gradWeight(x []float64, y float64) []float64 {
	pred := a.forwardProb(x)
	return scalarMul(-(pred-y)*pred*(1-pred), x)
}

func (a *Perceptron) gradBias(x []float64, y float64) float64 {
	pred := a.forwardProb(x)
	return -(pred - y) * pred * (1 - pred)
}

func (a *Perceptron) train() {
	for i := 0; i < a.epochs; i++ {
		dw := make([]float64, len(a.input[0]))
		db := 0.0
		for length, val := range a.input {
			dw = vecAdd(dw, a.gradWeight(val, a.Output[length]))
			db += a.gradBias(val, a.Output[length])
		}
		dw = scalarMul(2/float64(len(a.Output)), dw)
		a.weights = vecAdd(a.weights, dw)
		a.bias += db * 2 / float64(len(a.Output))
	}
}
