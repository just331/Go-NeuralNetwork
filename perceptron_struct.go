package Go_NeuralNetwork

/*
	Single-layer perceptron consists of:
		1.) Input layer - x
		2.) Output layer - y
		3.) Set of weights - w and bias - b
  		4.) Activation function - a for output layer. Sigmoid activation function
*/

type Perceptron struct {
	input   [][]float64
	Output  []float64
	weights []float64
	bias    float64
	epochs  int
}
